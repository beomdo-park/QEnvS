"""The key tree of randomised environments.

Every random draw of an episode is derived from the episode key alone, through a fixed
tree of ``jax.random.fold_in`` calls, so that changing how one leaf is used (how many
draws, which distribution, on or off) never changes the samples of another leaf::

    run_key
    episode_key[i] = fold_in(run_key, i)
        0 parameters             fold_in(episode_key, 0)
            one key per name     fold_in(., crc32(name))
        1 control errors         fold_in(episode_key, 1)
            0 calibration        fold_in(., 0)
                0 gain error     fold_in(., 0)
                1 offset         fold_in(., 1)
            1 step noise         fold_in(., 1)
                step k           fold_in(., k)
        2 measurement            fold_in(episode_key, 2)
            observation t        fold_in(., t)   (t = 0 after reset, t = k after step k)
                observable j     fold_in(., j)
        3 policy (evaluation)    fold_in(episode_key, 3)
            decision t           fold_in(., t)

Keys are a function of the episode key and the step index only, so restarts and parallel
runs do not depend on call order. Statistical independence between leaves is not proved
by this construction; the tests check sample distributions and correlations empirically.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import jax
from jax import Array

PARAMETERS = 0
CONTROL = 1
MEASUREMENT = 2
POLICY = 3

CALIBRATION = 0
STEP_NOISE = 1

GAIN = 0
OFFSET = 1


def episode_key(run_key: Array, i: int | Array) -> Array:
    """Key of the ``i``-th episode of a run (the rule used by vector envs and evaluation)."""
    return jax.random.fold_in(run_key, i)


def name_tag(name: str) -> int:
    """Stable 31-bit tag of a parameter name (CRC32, so it does not depend on the order)."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def check_name_tags(names: Iterable[str]) -> None:
    """Raise ``ValueError`` if two different names share a tag."""
    seen: dict[int, str] = {}
    for name in names:
        tag = name_tag(name)
        if tag in seen and seen[tag] != name:
            msg = f"parameter names {seen[tag]!r} and {name!r} share a key tag; rename one"
            raise ValueError(msg)
        seen[tag] = name


def parameter_key(ep_key: Array, name: str) -> Array:
    """Key of the model parameter ``name`` (fixed for the episode)."""
    return jax.random.fold_in(jax.random.fold_in(ep_key, PARAMETERS), name_tag(name))


def calibration_key(ep_key: Array, which: int) -> Array:
    """Key of an episode-fixed control error: ``GAIN`` or ``OFFSET``."""
    control = jax.random.fold_in(ep_key, CONTROL)
    return jax.random.fold_in(jax.random.fold_in(control, CALIBRATION), which)


def step_noise_key(ep_key: Array, k: int | Array) -> Array:
    """Key of the control noise of step ``k``."""
    control = jax.random.fold_in(ep_key, CONTROL)
    return jax.random.fold_in(jax.random.fold_in(control, STEP_NOISE), k)


def measurement_key(ep_key: Array, t: int | Array, j: int | Array) -> Array:
    """Key of the finite-shot estimate of observable ``j`` in observation ``t``."""
    meas = jax.random.fold_in(ep_key, MEASUREMENT)
    return jax.random.fold_in(jax.random.fold_in(meas, t), j)


def policy_key(ep_key: Array, t: int | Array) -> Array:
    """Key handed to a stochastic policy at decision ``t`` (evaluation only)."""
    return jax.random.fold_in(jax.random.fold_in(ep_key, POLICY), t)
