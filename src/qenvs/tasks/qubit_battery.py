"""Quantum battery tasks on the open two-level model."""

from __future__ import annotations

from qenvs.envs.base import Environment
from qenvs.envs.battery import BatteryCharging
from qenvs.models.qubit_battery import QubitBattery
from qenvs.registry import DIMENSIONLESS_UNITS, register

MODEL_RATE_KWARGS = ("detuning", "gamma1", "n_th", "gamma_phi")


def two_level_battery(**kw: object) -> Environment:
    """Build ``battery/TwoLevel-v0``: model kwargs to :class:`QubitBattery`, the rest to the env."""
    model_kw: dict[str, object] = {}
    for name in MODEL_RATE_KWARGS:
        if name in kw:
            model_kw[name] = float(kw.pop(name))  # type: ignore[arg-type]
    if "initial" in kw:
        model_kw["initial"] = str(kw.pop("initial"))
    model = QubitBattery(**model_kw)  # type: ignore[arg-type]
    return BatteryCharging(model=model, **kw)  # type: ignore[arg-type]


register(
    "battery/TwoLevel-v0",
    two_level_battery,
    description=(
        "Charge a qubit battery H_B = omega0 |1><1| coupled to a thermal bath (rate gamma1, Bose "
        "occupation n_th) with pure dephasing, using a piecewise-constant drive u(t) sigma_x/2. "
        "Reward per step = gain of stored energy (or ergotropy); return = Q_final - Q_initial. "
        "Closed optimum: any pulse with integral(u dt) = pi stores E = ergotropy = omega0."
    ),
    model="qenvs.models.qubit_battery.QubitBattery",
    references=(
        "Allahverdyan, Balian & Nieuwenhuizen, EPL 67, 565 (2004)",
        "Alicki & Fannes, Phys. Rev. E 87, 042123 (2013)",
        "Breuer & Petruccione, The Theory of Open Quantum Systems (2002), Sec. 3.4",
    ),
    units={
        **DIMENSIONLESS_UNITS,
        "control": "amplitude u multiplies sigma_x/2, Rabi frequency u (units of omega0)",
    },
    kwargs={
        "horizon": "int, control steps per episode (default 20)",
        "dt": "float, control period (default 0.1)",
        "action_bound": "float, |u| <= action_bound (default 2.0)",
        "reward": "'energy' or 'ergotropy', per-step gain of that quantity (default 'energy')",
        "detuning": "float, drive-qubit detuning Delta (default 0)",
        "gamma1": "float, bath coupling rate; >0 switches to Lindblad dynamics (default 0)",
        "n_th": "float, bath Bose occupation; emission gamma1(n_th+1), absorption gamma1 n_th "
        "(default 0)",
        "gamma_phi": "float, pure dephasing rate (default 0)",
        "initial": "'ground' or 'thermal' (Gibbs state p1 = n_th/(2 n_th + 1); needs gamma1 > 0 "
        "when n_th > 0) (default 'ground')",
    },
)
