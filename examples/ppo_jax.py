"""Vectorised PPO on a QEnvS environment with the functional JAX API (equinox + optax).

``num_envs`` environments are stepped in lock-step with ``jax.vmap(env.step)``, rollouts
are collected with ``lax.scan`` and the whole training run (rollout, GAE, clipped PPO
objective with entropy bonus, minibatch epochs) is one jitted ``lax.scan`` over updates.
Episodes have a fixed horizon; an environment is reset as soon as it reports ``done``.

The policy is a diagonal-Gaussian MLP (state-independent log standard deviation) and the
value function a separate MLP, both ``equinox`` modules; the environment clips actions to
``[-action_bound, action_bound]``.

The loop structure (vmapped environments, scan-based rollouts, GAE, minibatch epochs
inside a single jitted update) follows PureJaxRL by Chris Lu
(Apache-2.0, https://github.com/luchris429/purejaxrl). The code here is written from
scratch for the QEnvS functional API and equinox.

Usage::

    python examples/ppo_jax.py --task qubit/StateTransfer-v0 --num-envs 256 --updates 50 \
        --set horizon=10 --set dt=0.2 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

import qenvs
from qenvs.envs.base import EnvState

LOG_2PI = float(jnp.log(2.0 * jnp.pi))


def parse_set(items: Sequence[str] | None) -> dict[str, object]:
    """Parse ``--set key=value`` items; values are JSON-decoded when possible (as in ``qenvs``)."""
    out: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            msg = f"expected key=value, got {item!r}"
            raise argparse.ArgumentTypeError(msg)
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


class ActorCritic(eqx.Module):
    """Gaussian policy MLP + value MLP with a learned, state-independent ``log_std``."""

    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: Array

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, key: Array) -> None:
        k_actor, k_critic = jax.random.split(key)
        actor = eqx.nn.MLP(obs_dim, action_dim, hidden, 2, activation=jax.nn.tanh, key=k_actor)
        # small final layer so the initial mean action is close to zero (exploration from std=1)
        last = actor.layers[-1]
        self.actor = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            actor,
            (0.01 * last.weight, jnp.zeros_like(last.bias)),
        )
        self.critic = eqx.nn.MLP(obs_dim, "scalar", hidden, 2, activation=jax.nn.tanh, key=k_critic)
        self.log_std = jnp.zeros(action_dim)

    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Mean action and state value for one observation."""
        return self.actor(obs), self.critic(obs)


def gaussian_log_prob(mean: Array, log_std: Array, action: Array) -> Array:
    """Log density of a diagonal Gaussian, summed over the action dimension."""
    z = (action - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(z * z, axis=-1) - jnp.sum(log_std) - 0.5 * LOG_2PI * mean.shape[-1]


def gaussian_entropy(log_std: Array) -> Array:
    """Entropy of a diagonal Gaussian."""
    return jnp.sum(log_std + 0.5 * (LOG_2PI + 1.0))


class Transition(NamedTuple):
    """One vectorised environment step stored for the PPO update."""

    done: Array
    action: Array
    value: Array
    reward: Array
    log_prob: Array
    obs: Array
    episode_return: Array  # return of the episode that ended at this step (0 otherwise)


def _select(done: Array, on_done: object, otherwise: object) -> object:
    """``where(done, on_done, otherwise)`` over a pytree with a leading batch axis."""

    def pick(a: Array, b: Array) -> Array:
        d = done.reshape(done.shape + (1,) * (b.ndim - 1))
        return jnp.where(d, a, b)

    return jax.tree_util.tree_map(pick, on_done, otherwise)


def run_ppo(
    task: str,
    num_envs: int = 256,
    updates: int = 50,
    seed: int = 0,
    *,
    num_steps: int | None = None,
    lr: float = 1e-3,
    epochs: int = 4,
    minibatches: int = 4,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    hidden: int = 64,
    eval_envs: int | None = None,
    **task_kwargs: object,
) -> dict[str, object]:
    """Train PPO on ``task`` and return a JSON-serialisable summary.

    Args:
        task: registered task id, e.g. ``"qubit/StateTransfer-v0"``.
        num_envs: number of environments stepped in parallel.
        updates: number of PPO updates (each collects ``num_steps * num_envs`` transitions).
        seed: PRNG seed for network initialisation, action noise and minibatch shuffling.
        num_steps: environment steps per update (default: the task horizon, one episode).
        lr: Adam learning rate.
        epochs: passes over the collected batch per update.
        minibatches: minibatches per epoch (must divide ``num_steps * num_envs``).
        discount: discount factor. (Named ``discount`` rather than ``gamma`` so it can never
            capture a task kwarg: ``battery/Dicke-v0`` has a decay rate called ``gamma``.)
        gae_lambda: GAE lambda.
        clip_eps: PPO clipping parameter (policy ratio and value).
        ent_coef: entropy bonus coefficient.
        vf_coef: value-loss coefficient.
        max_grad_norm: global gradient-norm clipping.
        hidden: hidden width of the two-layer MLPs.
        eval_envs: environments used for the final deterministic evaluation (default ``num_envs``).
        **task_kwargs: forwarded to ``qenvs.make`` (``horizon``, ``dt``, ``gamma1``, ...).

    Returns:
        Dictionary with ``final_return`` (mean deterministic-policy return over ``eval_envs``
        episodes), ``final_return_std``, ``random_return`` (uniform random actions, same
        episodes), ``learning_curve`` (mean episodic return of the stochastic policy per
        update), ``wall_time_s`` and ``compile_time_s``.
    """
    env = qenvs.make(task, **task_kwargs)
    horizon = int(env.horizon)
    num_steps = horizon if num_steps is None else int(num_steps)
    eval_envs = num_envs if eval_envs is None else int(eval_envs)
    batch_size = num_steps * num_envs
    if updates < 1:
        msg = f"updates must be >= 1, got {updates}"
        raise ValueError(msg)
    if batch_size % minibatches != 0:
        msg = f"num_steps * num_envs = {batch_size} must be divisible by minibatches={minibatches}"
        raise ValueError(msg)
    obs_dim, action_dim = int(env.observation_dim), int(env.action_dim)
    bound = float(env.action_bound)

    key = jax.random.PRNGKey(seed)
    key, k_model = jax.random.split(key)
    model = ActorCritic(obs_dim, action_dim, hidden, k_model)
    params, static = eqx.partition(model, eqx.is_array)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def policy(p: object, obs: Array) -> tuple[Array, Array, Array]:
        m = eqx.combine(p, static)
        mean, value = jax.vmap(m)(obs)
        return mean, m.log_std, value

    def env_step(carry: tuple, _: None) -> tuple[tuple, Transition]:
        p, env_state, last_obs, ep_return, key = carry
        key, k_act, k_reset = jax.random.split(key, 3)
        mean, log_std, value = policy(p, last_obs)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape, mean.dtype)
        log_prob = gaussian_log_prob(mean, log_std, action)
        stepped, ts = v_step(env_state, action)
        done = ts.done
        ep_return = ep_return + ts.reward
        reset_state, reset_ts = v_reset(jax.random.split(k_reset, num_envs))
        env_state = _select(done, reset_state, stepped)
        obs = jnp.where(done[:, None], reset_ts.observation, ts.observation)
        tr = Transition(
            done, action, value, ts.reward, log_prob, last_obs, jnp.where(done, ep_return, 0.0)
        )
        ep_return = jnp.where(done, 0.0, ep_return)
        return (p, env_state, obs, ep_return, key), tr

    def gae(traj: Transition, last_value: Array) -> tuple[Array, Array]:
        def body(carry: tuple[Array, Array], tr: Transition) -> tuple[tuple[Array, Array], Array]:
            adv, next_value = carry
            not_done = 1.0 - tr.done.astype(adv.dtype)
            delta = tr.reward + discount * next_value * not_done - tr.value
            adv = delta + discount * gae_lambda * not_done * adv
            return (adv, tr.value), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        return adv, adv + traj.value

    def loss_fn(p: object, tr: Transition, adv: Array, target: Array) -> tuple[Array, dict]:
        mean, log_std, value = policy(p, tr.obs)
        log_prob = gaussian_log_prob(mean, log_std, tr.action)
        ratio = jnp.exp(log_prob - tr.log_prob)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.minimum(
            ratio * adv_n, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n
        ).mean()
        v_clipped = tr.value + jnp.clip(value - tr.value, -clip_eps, clip_eps)
        v_loss = 0.5 * jnp.maximum((value - target) ** 2, (v_clipped - target) ** 2).mean()
        entropy = gaussian_entropy(log_std)
        total = pg_loss + vf_coef * v_loss - ent_coef * entropy
        aux = {"policy_loss": pg_loss, "value_loss": v_loss, "entropy": entropy}
        return total, aux

    def update_minibatch(carry: tuple, batch: tuple) -> tuple[tuple, dict]:
        p, opt_state = carry
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, *batch)
        upd, opt_state = tx.update(grads, opt_state, p)
        return (optax.apply_updates(p, upd), opt_state), aux

    def update_epoch(carry: tuple, _: None) -> tuple[tuple, dict]:
        p, opt_state, key, batch = carry
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, batch_size)

        def shuffle(x: Array) -> Array:
            x = x.reshape((batch_size, *x.shape[2:]))
            return jnp.take(x, perm, axis=0).reshape((minibatches, -1, *x.shape[1:]))

        (p, opt_state), aux = jax.lax.scan(
            update_minibatch, (p, opt_state), jax.tree_util.tree_map(shuffle, batch)
        )
        return (p, opt_state, key, batch), aux

    def update_step(runner: tuple, _: None) -> tuple[tuple, dict[str, Array]]:
        p, opt_state, env_state, last_obs, ep_return, key = runner
        carry, traj = jax.lax.scan(
            env_step, (p, env_state, last_obs, ep_return, key), None, num_steps
        )
        _, env_state, last_obs, ep_return, key = carry
        _, _, last_value = policy(p, last_obs)
        adv, target = gae(traj, last_value)
        (p, opt_state, key, _), aux = jax.lax.scan(
            update_epoch, (p, opt_state, key, (traj, adv, target)), None, epochs
        )
        n_done = traj.done.sum()
        metrics = {
            "episode_return": traj.episode_return.sum() / jnp.maximum(n_done, 1),
            "episodes": n_done,
            **{k: v.mean() for k, v in aux.items()},
        }
        return (p, opt_state, env_state, last_obs, ep_return, key), metrics

    @jax.jit
    def train(p: object, key: Array) -> tuple[object, dict[str, Array]]:
        key, k_reset = jax.random.split(key)
        env_state, ts = v_reset(jax.random.split(k_reset, num_envs))
        ep_return = jnp.zeros(num_envs, ts.reward.dtype)
        runner = (p, tx.init(p), env_state, ts.observation, ep_return, key)
        runner, metrics = jax.lax.scan(update_step, runner, None, updates)
        return runner[0], metrics

    @jax.jit
    def evaluate(p: object, key: Array) -> tuple[Array, Array, Array]:
        """Deterministic policy (mean action) vs uniform random actions, same ``eval_envs`` keys."""
        keys = jax.random.split(key, eval_envs)
        state0, ts0 = v_reset(keys)

        def det_body(carry: tuple[EnvState, Array], _: None) -> tuple[tuple, tuple]:
            state, obs = carry
            mean, _, _ = policy(p, obs)
            state, ts = v_step(state, mean)
            return (state, ts.observation), (ts.reward, mean[0])

        def rand_body(carry: tuple[EnvState, Array], _: None) -> tuple[tuple, Array]:
            state, k = carry
            k, k_a = jax.random.split(k)
            a = jax.random.uniform(k_a, (eval_envs, action_dim), minval=-bound, maxval=bound)
            state, ts = v_step(state, a)
            return (state, k), ts.reward

        _, (det_rewards, det_actions) = jax.lax.scan(
            det_body, (state0, ts0.observation), None, horizon
        )
        _, rand_rewards = jax.lax.scan(rand_body, (state0, key), None, horizon)
        return det_rewards.sum(0), rand_rewards.sum(0), det_actions

    t0 = time.perf_counter()
    key, k_train, k_eval = jax.random.split(key, 3)
    train_compiled = train.lower(params, k_train).compile()
    compile_time = time.perf_counter() - t0
    params, metrics = train_compiled(params, k_train)
    jax.block_until_ready(metrics)
    train_time = time.perf_counter() - t0 - compile_time
    det_returns, rand_returns, det_actions = evaluate(params, k_eval)
    jax.block_until_ready(det_returns)
    wall = time.perf_counter() - t0

    curve = [float(x) for x in metrics["episode_return"]]
    return {
        "task": task,
        "task_kwargs": dict(task_kwargs),
        "method": "ppo",
        "num_envs": num_envs,
        "updates": updates,
        "num_steps": num_steps,
        "seed": seed,
        "hyperparameters": {
            "lr": lr,
            "epochs": epochs,
            "minibatches": minibatches,
            "discount": discount,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "hidden": hidden,
        },
        "horizon": horizon,
        "dt": float(env.dt),
        "action_dim": action_dim,
        "observation_dim": obs_dim,
        "action_bound": bound,
        "open_system": bool(env.model.is_open),
        "transitions": updates * batch_size,
        "initial_return": curve[0],
        "final_return": float(det_returns.mean()),
        "final_return_std": float(det_returns.std()),
        # deterministic (mean-action) pulse of the first evaluation env, clipped as in env.step
        "policy_pulse": [[float(x) for x in jnp.clip(row, -bound, bound)] for row in det_actions],
        "random_return": float(rand_returns.mean()),
        "learning_curve": curve,
        "final_entropy": float(metrics["entropy"][-1]),
        "final_log_std": [float(x) for x in jnp.atleast_1d(params.log_std)],
        "wall_time_s": wall,
        "compile_time_s": compile_time,
        "train_time_s": train_time,
        "eval_envs": eval_envs,
    }


def build_parser() -> argparse.ArgumentParser:
    """Command line options (mirrors ``examples/grape_jax.py``)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="qubit/StateTransfer-v0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="task kwargs, e.g. --set gamma1=0.1"
    )
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point (enables JAX x64 like the ``qenvs`` CLI)."""
    jax.config.update("jax_enable_x64", True)
    args = build_parser().parse_args(argv)
    out = run_ppo(
        args.task, args.num_envs, args.updates, args.seed, lr=args.lr, **parse_set(args.set)
    )
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        curve = out["learning_curve"]  # type: ignore[assignment]
        shown = ", ".join(f"{x:.3f}" for x in curve[:: max(1, len(curve) // 10)])  # type: ignore[index]
        sys.stdout.write(
            f"{out['task']} num_envs={out['num_envs']} updates={out['updates']} "
            f"seed={out['seed']} horizon={out['horizon']} dt={out['dt']}\n"
            f"deterministic return: {out['final_return']:.4f} +- {out['final_return_std']:.4f} "
            f"(random actions: {out['random_return']:.4f})\n"
            f"learning curve (stochastic policy, every {max(1, len(curve) // 10)} updates): "
            f"[{shown}]\n"
            f"wall time: {out['wall_time_s']:.2f} s\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
