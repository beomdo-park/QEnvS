"""Three-level Lambda-system tasks (STIRAP-style state transfer)."""

from __future__ import annotations

import dynamiqs as dq

from qenvs.envs.base import Environment
from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.lambda_system import LambdaSystem, stirap_pulses
from qenvs.registry import DIMENSIONLESS_UNITS, register


def _lambda_stirap(**kw: object) -> Environment:
    delta_one_photon = float(kw.pop("delta_one_photon", 0.0))
    delta_two_photon = float(kw.pop("delta_two_photon", 0.0))
    gamma = float(kw.pop("gamma", 0.0))
    model = LambdaSystem(
        delta_one_photon=delta_one_photon, delta_two_photon=delta_two_photon, gamma=gamma
    )
    target = dq.basis(3, 2)  # pure target kept as a ket: exact, differentiable fidelity
    kw.setdefault("horizon", 100)
    kw.setdefault("dt", 0.2)
    kw.setdefault("action_bound", 1.0)
    return StateTransfer(model=model, target=target, **kw)  # type: ignore[arg-type]


def stirap_reference(env: Environment) -> object:
    """Counter-intuitive Gaussian pulse pair scaled to the episode length ``T``.

    Width ``0.3 T`` and separation ``tau = 0.15 T`` at the full Rabi bound; on the default
    task (T = 20) this transfers 0.9964 of the population to |3>.
    """
    t = env.horizon * env.dt
    return stirap_pulses(env.horizon, env.dt, env.action_bound, 0.15 * t, 0.3 * t)


register(
    "transfer/LambdaSTIRAP-v0",
    _lambda_stirap,
    description=(
        "Transfer a three-level Lambda system from |1> to |3> through the radiatively decaying "
        "state |2> with piecewise-constant pump Omega_p(t) (|1>-|2>) and Stokes Omega_s(t) "
        "(|3>-|2>) Rabi frequencies; action = (Omega_p, Omega_s). Reward per step = fidelity "
        "gain to |3>; return = F_final - F_initial. Reference solution: STIRAP, i.e. Stokes "
        "before pump (counter-intuitive order) with Omega_0 T >> 1 keeps the state in the dark "
        "state Omega_s|1> - Omega_p|3> and never populates |2>; see "
        "qenvs.models.lambda_system.stirap_pulses."
    ),
    model="qenvs.models.lambda_system.LambdaSystem",
    references=(
        "Bergmann, Theuer & Shore, Rev. Mod. Phys. 70, 1003 (1998)",
        "Vitanov, Rangelov, Shore & Bergmann, Rev. Mod. Phys. 89, 015006 (2017)",
        "Shore, Acta Physica Slovaca 58, 243 (2008)",
    ),
    reference=stirap_reference,
    units={
        **DIMENSIONLESS_UNITS,
        "control": (
            "pump and Stokes Rabi frequencies (Omega_p, Omega_s) in units of the inverse time unit"
        ),
    },
    kwargs={
        "horizon": "int, control steps per episode (default 100)",
        "dt": "float, control period (default 0.2, so T = 20)",
        "action_bound": (
            "float, |Omega_p|, |Omega_s| <= action_bound, i.e. the maximum Rabi frequency "
            "(default 1.0)"
        ),
        "delta_one_photon": "float, one-photon (pump) detuning Delta of |2> (default 0)",
        "delta_two_photon": "float, two-photon detuning delta of |3> (default 0)",
        "gamma": (
            "float, spontaneous-emission rate of |2>, equal branching into |1> and |3>; "
            ">0 switches to Lindblad dynamics (default 0)"
        ),
    },
)
