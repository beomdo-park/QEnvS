"""Dicke / Tavis-Cummings battery task with local (per-cell) dissipation."""

from __future__ import annotations

from qenvs.envs.base import Environment
from qenvs.envs.battery import BatteryCharging
from qenvs.models.dicke_local import MAX_HILBERT_DIM, MAX_LIOUVILLIAN_DIM, DickeBatteryLocal
from qenvs.registry import DIMENSIONLESS_UNITS, register

DEFAULT_GAMMA_LOCAL = 0.05
DEFAULT_GAMMA_PHI_LOCAL = 0.05


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _dicke_local_battery(**kw: object) -> Environment:
    model = DickeBatteryLocal(
        n_cells=int(kw.pop("n_cells", 3)),
        n_photons=_optional_int(kw.pop("n_photons", None)),
        n_max=_optional_int(kw.pop("n_max", None)),
        omega0=float(kw.pop("omega0", 1.0)),
        omega_c=float(kw.pop("omega_c", 1.0)),
        rwa=bool(kw.pop("rwa", True)),
        kappa=float(kw.pop("kappa", 0.0)),
        gamma=float(kw.pop("gamma", 0.0)),
        gamma_phi=float(kw.pop("gamma_phi", 0.0)),
        gamma_local=float(kw.pop("gamma_local", DEFAULT_GAMMA_LOCAL)),
        gamma_phi_local=float(kw.pop("gamma_phi_local", DEFAULT_GAMMA_PHI_LOCAL)),
        pumping_local=float(kw.pop("pumping_local", 0.0)),
    )
    kw.setdefault("horizon", 20)
    kw.setdefault("dt", 0.2)
    kw.setdefault("action_bound", 0.5)
    return BatteryCharging(model=model, **kw)  # type: ignore[arg-type]


register(
    "battery/DickeLocal-v0",
    _dicke_local_battery,
    description=(
        "Charge N two-level cells from a cavity initially holding n_photons photons by "
        "controlling the collective coupling g(t) (Tavis-Cummings for rwa=True, Dicke for "
        "rwa=False), as in battery/Dicke-v0, but with local (per-cell) emission, dephasing and "
        "pumping that drive the cells out of the permutation-symmetric subspace. Solved exactly "
        "per control step (dense expm of the Liouvillian) with the permutation-invariant method "
        "of Shammah et al. 2018 over all total-spin sectors j = N/2, N/2 - 1, ...; the state is "
        "the full 2^N x cavity density matrix in the Schur basis. Reward per step = gain of the "
        "stored energy <H_B> (reward='energy') or of the ergotropy of the full 2^N-dimensional "
        "cell state, cavity traced out (reward='ergotropy'); return = Q_final - Q_initial. This "
        "ergotropy allows any unitary on the 2^N cell space and is therefore larger than or "
        "equal to the symmetric-subspace ergotropy reported by battery/Dicke-v0. With every "
        "local rate at zero the energy dynamics equal battery/Dicke-v0. The default Fock cutoff "
        "n_photons is exact for rwa=True without pumping; with pumping_local > 0 pass n_max and "
        "check the top_fock population in info. Size limit: Liouvillian "
        f"dimension sum_j ((2j+1)(n_max+1))^2 <= {MAX_LIOUVILLIAN_DIM} and 2^N (n_max+1) <= "
        f"{MAX_HILBERT_DIM} (N <= 5 with n_max = N); one step costs about 0.07 s at the "
        "default N=3 and 1 s at N=5, n_max=5 on a CPU."
    ),
    model="qenvs.models.dicke_local.DickeBatteryLocal",
    references=(
        "Shammah, Ahmed, Lambert, De Liberato, Nori, Phys. Rev. A 98, 063815 (2018), "
        "arXiv:1805.05129",
        "Ferraro, Campisi, Andolina, Pellegrini, Polini, Phys. Rev. Lett. 120, 117702 (2018)",
        "Tavis & Cummings, Phys. Rev. 170, 379 (1968)",
        "Dicke, Phys. Rev. 93, 99 (1954)",
        "Allahverdyan, Balian & Nieuwenhuizen, EPL 67, 565 (2004)",
    ),
    units={
        **DIMENSIONLESS_UNITS,
        "control": "collective coupling g(t) in units of omega0",
        "rates": (
            "kappa, gamma, gamma_phi (collective) and gamma_local, gamma_phi_local, "
            "pumping_local (per cell) in units of omega0"
        ),
    },
    kwargs={
        "n_cells": "int, number of two-level cells N (default 3)",
        "n_photons": "int, initial cavity Fock number (default n_cells)",
        "n_max": (
            "int, Fock cutoff (default n_photons if rwa, exact only when pumping_local = 0; "
            "n_photons + 4 if not rwa). Required when pumping_local > 0: the excitation number "
            "is then unbounded and no finite cutoff is exact, so raise n_max until the top_fock "
            "population in info is negligible"
        ),
        "omega0": "float, cell transition frequency, energy unit of H_B (default 1.0)",
        "omega_c": "float, cavity frequency; detuning omega_c - omega0 (default 1.0, resonant)",
        "rwa": "bool, True = Tavis-Cummings, False = full Dicke with counter-rotating terms",
        "kappa": "float, cavity photon loss rate sqrt(kappa) a (default 0)",
        "gamma": "float, collective emission rate sqrt(gamma) J- (default 0)",
        "gamma_phi": "float, collective dephasing rate sqrt(gamma_phi) Jz (default 0)",
        "gamma_local": (
            f"float, per-cell emission rate sqrt(gamma_local) sigma_- on every cell "
            f"(default {DEFAULT_GAMMA_LOCAL})"
        ),
        "gamma_phi_local": (
            "float, per-cell dephasing rate sqrt(gamma_phi_local) sigma_z / 2 on every cell; a "
            "single-cell coherence decays at gamma_phi_local / 2 "
            f"(default {DEFAULT_GAMMA_PHI_LOCAL})"
        ),
        "pumping_local": (
            "float, per-cell incoherent pumping rate sqrt(pumping_local) sigma_+ (default 0)"
        ),
        "horizon": "int, control steps per episode (default 20)",
        "dt": "float, control period in units of 1/omega0 (default 0.2)",
        "action_bound": "float, |g| <= action_bound = g_max (default 0.5)",
        "reward": "'energy' or 'ergotropy' of the full 2^N cell state (default 'energy')",
    },
)
