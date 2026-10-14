"""Collective (Dicke / Tavis-Cummings) battery tasks."""

from __future__ import annotations

from qenvs.envs.base import Environment
from qenvs.envs.battery import BatteryCharging
from qenvs.models.dicke import DickeBattery
from qenvs.registry import DIMENSIONLESS_UNITS, register


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _dicke_battery(**kw: object) -> Environment:
    model = DickeBattery(
        n_cells=int(kw.pop("n_cells", 3)),
        n_photons=_optional_int(kw.pop("n_photons", None)),
        n_max=_optional_int(kw.pop("n_max", None)),
        omega0=float(kw.pop("omega0", 1.0)),
        omega_c=float(kw.pop("omega_c", 1.0)),
        rwa=bool(kw.pop("rwa", True)),
        kappa=float(kw.pop("kappa", 0.0)),
        gamma=float(kw.pop("gamma", 0.0)),
        gamma_phi=float(kw.pop("gamma_phi", 0.0)),
    )
    kw.setdefault("horizon", 20)
    kw.setdefault("dt", 0.2)
    kw.setdefault("action_bound", 0.5)
    return BatteryCharging(model=model, **kw)  # type: ignore[arg-type]


register(
    "battery/Dicke-v0",
    _dicke_battery,
    description=(
        "Charge N two-level cells from a cavity initially holding n_photons photons (Ferraro "
        "et al. 2018) by controlling the collective coupling g(t) in the Tavis-Cummings "
        "(rwa=True) or Dicke (rwa=False) Hamiltonian, within the permutation-symmetric "
        "subspace. Reward per step = gain of the stored energy <H_B> (reward='energy') or of "
        "the ergotropy of the reduced cell state, cavity traced out (reward='ergotropy'); "
        "return = Q_final - Q_initial, Q <= N omega0. Constant g charges in a time that "
        "decreases with N (collective advantage): the fitted exponent in this protocol is "
        "about -0.40 for N <= 48 (Ferraro et al. state N^(-1/2); not established here). "
        "Switching g to 0 holds the charge. WARNING: for rwa=False the default Fock cutoff "
        "n_max = n_photons + 4 is NOT converged at the default action_bound = 0.5, and the "
        "needed cutoff grows with N (N=3: return 1.75 vs 1.53 converged at n_photons + 24; "
        "N=6 needs about n_photons + 56). Raise n_max until the 'top_fock' observable stays "
        "below about 1e-4 over the episode, or keep action_bound <= 0.1 for N <= 4."
    ),
    model="qenvs.models.DickeBattery",
    references=(
        "Ferraro, Campisi, Andolina, Pellegrini, Polini, Phys. Rev. Lett. 120, 117702 (2018)",
        "Tavis & Cummings, Phys. Rev. 170, 379 (1968)",
        "Dicke, Phys. Rev. 93, 99 (1954)",
        "Andolina, Keck, Mari, Giovannetti, Polini, Phys. Rev. B 99, 205437 (2019)",
        "Erdman, Andolina, Giovannetti, Noé, Phys. Rev. Lett. 133, 243602 (2024), arXiv:2212.12397",
        "Shammah, Ahmed, Lambert, De Liberato, Nori, Phys. Rev. A 98, 063815 (2018)",
    ),
    units={**DIMENSIONLESS_UNITS, "control": "collective coupling g(t) in units of omega0"},
    kwargs={
        "n_cells": "int, number of two-level cells N (default 3)",
        "n_photons": "int, initial cavity Fock number (default n_cells)",
        "n_max": (
            "int, Fock cutoff (default n_photons if rwa else n_photons + 4; for rwa=False "
            "converge it with the top_fock observable, see the task description)"
        ),
        "omega0": "float, cell transition frequency, energy unit of H_B (default 1.0)",
        "omega_c": "float, cavity frequency; detuning omega_c - omega0 (default 1.0, resonant)",
        "rwa": "bool, True = Tavis-Cummings, False = full Dicke with counter-rotating terms",
        "kappa": "float, cavity photon loss rate; >0 switches to Lindblad dynamics (default 0)",
        "gamma": "float, collective emission rate sqrt(gamma) J- (default 0)",
        "gamma_phi": "float, collective dephasing rate sqrt(gamma_phi) Jz (default 0)",
        "horizon": "int, control steps per episode (default 20)",
        "dt": "float, control period in units of 1/omega0 (default 0.2)",
        "action_bound": "float, |g| <= action_bound = g_max (default 0.5)",
        "reward": "'energy' or 'ergotropy' of the reduced cell state (default 'energy')",
    },
)
