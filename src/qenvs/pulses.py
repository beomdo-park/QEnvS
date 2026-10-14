"""Pulse I/O and device parameters for hardware toolchains.

A :class:`Pulse` is an immutable record of a piecewise-constant pulse as QEnvS produces
it (the actions of a rollout): the samples, the sample period, the channel names, the
units and the provenance. It serialises losslessly to JSON and NPZ, converts transmon
amplitudes between rad/ns and MHz, refines a pulse exactly onto a finer sample grid
(for example 2/9 ns) and pads it with zeros to the granularity of an arbitrary waveform
generator.

UNITS: ``dt_ns`` is the sample period in nanoseconds for the transmon tasks (units
``'rad/ns'``, amplitudes are angular frequencies Omega, the Rabi frequency is
``Omega / (2 pi)``). For the tasks in units of the transition frequency (units
``'dimensionless'``) the field keeps the name ``dt_ns`` but holds the dimensionless
period in units of ``1 / omega0``; the MHz helpers refuse such pulses.

:func:`parse_properties` reads device parameters (frequency, anharmonicity, T1, T2) from a
plain mapping with explicit units or with the SI values that Qiskit's
``BackendProperties.qubit_property`` returned; it is the engine behind
:meth:`qenvs.models.transmon.Transmon.from_properties`. Nothing here imports Qiskit.

QEnvS does not model an AWG transfer function, does not calibrate amplitudes and does not
talk to hardware; see ``docs/guides/pulses.md``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

from qenvs.registry import registry

SCHEMA = "qenvs.pulse/v1"
TWO_PI = 2.0 * math.pi

#: Units string of the transmon tasks (angular frequency in rad/ns).
UNITS_RAD_PER_NS = "rad/ns"
#: Units string of the tasks in units of the transition frequency omega0.
UNITS_DIMENSIONLESS = "dimensionless"

#: Relative tolerance on the ratio ``dt / sample_dt`` for :meth:`Pulse.resample`.
RESAMPLE_RTOL = 1e-9

PathOrFile = str | os.PathLike[str] | IO[bytes]


# ----------------------------------------------------------------------- units
def rad_per_ns_to_mhz(omega: Any) -> Any:
    """Rabi frequency in MHz of an angular frequency ``omega`` in rad/ns: ``omega / (2 pi) * 1e3``.

    A constant ``Omega = pi / 20`` rad/ns for 20 ns is a pi pulse and gives 25 MHz.
    """
    return np.asarray(omega, dtype=np.float64) / TWO_PI * 1e3


def mhz_to_rad_per_ns(rabi_mhz: Any) -> Any:
    """Angular frequency in rad/ns of a Rabi frequency in MHz: ``2 pi * rabi_mhz * 1e-3``."""
    return np.asarray(rabi_mhz, dtype=np.float64) * 1e-3 * TWO_PI


# ------------------------------------------------------------------- metadata
def _jsonable(obj: Any, path: str) -> Any:
    """Deep-copy ``obj`` into plain JSON types (dict, list, str, int, float, bool, None)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            msg = f"{path} is not finite ({x}); JSON has no representation for it"
            raise ValueError(msg)
        return x
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                msg = f"{path} has a non-string key {k!r}; JSON object keys must be strings"
                raise TypeError(msg)
            out[k] = _jsonable(v, f"{path}[{k!r}]")
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if hasattr(obj, "__array__"):  # numpy and jax arrays
        return _jsonable(np.asarray(obj).tolist(), path)
    msg = (
        f"{path} has type {type(obj).__name__}, which is not JSON serialisable; "
        "convert it to str, number, list or dict first"
    )
    raise TypeError(msg)


def _task_units(task_id: str) -> tuple[str, str]:
    """Return ``(units, model)`` of a registered task.

    A task whose ``units['time']`` starts with ``'ns'`` works in hardware units (angular
    frequency in rad/ns); every other task is in units of the transition frequency.
    """
    try:
        spec = registry[task_id]
    except KeyError as e:
        known = ", ".join(sorted(registry))
        msg = f"unknown task id {task_id!r}; known: {known}"
        raise KeyError(msg) from e
    hardware = spec.units.get("time", "").lstrip().startswith("ns")
    return (UNITS_RAD_PER_NS if hardware else UNITS_DIMENSIONLESS), spec.model


# ----------------------------------------------------------------------- Pulse
class Pulse:
    """Immutable piecewise-constant pulse.

    Args:
        samples: amplitudes, shape ``(n, channels)``; copied to a read-only float64 array.
        dt_ns: sample period (ns for ``'rad/ns'`` pulses), positive and finite.
        channels: one unique name per column; default ``'u0'``, ``'u1'``, ...
        units: ``'rad/ns'`` (transmon tasks), ``'dimensionless'`` (omega0-unit tasks) or any
            other label; only ``'rad/ns'`` supports the MHz helpers.
        metadata: JSON-compatible mapping (task id, task kwargs, qenvs version, fidelity,
            ...); deep-copied, so later changes of the argument do not leak in.

    Attributes:
        samples: read-only ``(n, channels)`` float64 array.
        metadata: a fresh deep copy on every access (the record itself cannot change).
    """

    __slots__ = ("_channels", "_dt_ns", "_meta", "_samples", "_units")

    def __init__(
        self,
        samples: Any,
        dt_ns: float,
        channels: Sequence[str] | None = None,
        units: str = UNITS_RAD_PER_NS,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if np.iscomplexobj(samples):
            msg = (
                "samples must be real: complex samples would lose their imaginary part; "
                "pass the two quadratures as two real channels (I, Q)"
            )
            raise TypeError(msg)
        arr = np.array(samples, dtype=np.float64, copy=True)
        if arr.ndim != 2:
            msg = (
                f"samples must have shape (n_samples, n_channels), got shape {arr.shape}; "
                "reshape a single channel to (n, 1)"
            )
            raise ValueError(msg)
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            msg = f"samples must have at least one sample and one channel, got shape {arr.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(arr)):
            msg = "samples must be finite"
            raise ValueError(msg)
        dt = float(dt_ns)
        if not (math.isfinite(dt) and dt > 0):
            msg = f"dt_ns must be positive and finite, got {dt_ns!r}"
            raise ValueError(msg)
        if channels is None:
            names = tuple(f"u{k}" for k in range(arr.shape[1]))
        else:
            if isinstance(channels, str):
                msg = f"channels must be a sequence of names, got the string {channels!r}"
                raise TypeError(msg)
            names = tuple(str(c) for c in channels)
        if len(names) != arr.shape[1] or len(set(names)) != len(names):
            msg = f"channels must be {arr.shape[1]} unique names (one per column), got {names!r}"
            raise ValueError(msg)
        if not isinstance(units, str) or not units:
            msg = f"units must be a non-empty string, got {units!r}"
            raise ValueError(msg)
        meta = _jsonable({} if metadata is None else dict(metadata), "metadata")
        arr.setflags(write=False)
        object.__setattr__(self, "_samples", arr)
        object.__setattr__(self, "_dt_ns", dt)
        object.__setattr__(self, "_channels", names)
        object.__setattr__(self, "_units", units)
        object.__setattr__(self, "_meta", json.dumps(meta, sort_keys=True, allow_nan=False))

    def __setattr__(self, name: str, value: object) -> None:
        msg = f"Pulse is immutable; cannot set {name!r} (use resample, pad or the constructors)"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        msg = f"Pulse is immutable; cannot delete {name!r}"
        raise AttributeError(msg)

    # ------------------------------------------------------------ properties
    @property
    def samples(self) -> np.ndarray:
        """Read-only ``(n, channels)`` float64 array."""
        return self._samples

    @property
    def dt_ns(self) -> float:
        """Sample period (ns for ``'rad/ns'`` pulses)."""
        return self._dt_ns

    @property
    def channels(self) -> tuple[str, ...]:
        """Channel names, one per column."""
        return self._channels

    @property
    def units(self) -> str:
        """Units of the amplitudes: ``'rad/ns'`` or ``'dimensionless'``."""
        return self._units

    @property
    def metadata(self) -> dict[str, Any]:
        """Provenance and optional numbers; a fresh deep copy on every access."""
        return json.loads(self._meta)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self._samples.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self._samples.shape[1])

    @property
    def duration_ns(self) -> float:
        """Total duration ``n_samples * dt_ns``."""
        return self.n_samples * self._dt_ns

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pulse):
            return NotImplemented
        return (
            self._dt_ns == other._dt_ns
            and self._channels == other._channels
            and self._units == other._units
            and np.array_equal(self._samples, other._samples)
            and self.metadata == other.metadata
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Pulse(n_samples={self.n_samples}, channels={self._channels}, "
            f"dt_ns={self._dt_ns!r}, units={self._units!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (Pulse.from_dict, (self.to_dict(),))

    def _replace(
        self,
        samples: np.ndarray | None = None,
        dt_ns: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Pulse:
        return Pulse(
            self._samples if samples is None else samples,
            self._dt_ns if dt_ns is None else dt_ns,
            self._channels,
            self._units,
            self.metadata if metadata is None else metadata,
        )

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_actions(
        cls,
        actions: Any,
        dt: float,
        *,
        task_id: str | None = None,
        task_kwargs: Mapping[str, Any] | None = None,
        channels: Sequence[str] | None = None,
        units: str | None = None,
        **metadata: Any,
    ) -> Pulse:
        """Record the actions of a rollout (shape ``(horizon, action_dim)``) as a pulse.

        Args:
            actions: numpy or jax array of actions, ``(horizon, action_dim)``.
            dt: the control period of the environment (``env.dt``).
            task_id: registered task id; sets the units from ``registry[task_id].units``
                (``'rad/ns'`` when its time unit is ns, else ``'dimensionless'``) and the
                default channel names (``('I', 'Q')`` for a Transmon model with two
                controls, else ``'u0'``, ``'u1'``, ...).
            task_kwargs: kwargs the task was made with, kept as provenance (must be JSON
                compatible).
            channels: explicit channel names.
            units: explicit units; defaults to the task's, or ``'rad/ns'`` without a task id
                (the hardware units of the transmon tasks, assumed, not checked). Giving
                units that contradict the task raises ``ValueError``.
            **metadata: extra JSON-compatible provenance, e.g. ``fidelity=0.9999``.

        The metadata always records the ``qenvs_version``, plus ``task_id`` and
        ``task_kwargs`` when given. Float32 actions are widened to float64 exactly.
        """
        from qenvs import __version__

        arr = np.asarray(actions, dtype=np.float64)
        meta: dict[str, Any] = {"qenvs_version": __version__}
        model = ""
        task_units = None
        if task_id is not None:
            task_units, model = _task_units(task_id)
            meta["task_id"] = task_id
        if task_kwargs is not None:
            meta["task_kwargs"] = dict(task_kwargs)
        meta.update(metadata)
        if units is not None and task_units is not None and units != task_units:
            msg = f"units={units!r} contradicts task {task_id!r}, whose units are {task_units!r}"
            raise ValueError(msg)
        final_units = units or task_units or UNITS_RAD_PER_NS
        if channels is None and model.endswith(".Transmon") and arr.ndim == 2 and arr.shape[1] == 2:
            channels = ("I", "Q")
        return cls(arr, dt, channels, final_units, meta)

    @classmethod
    def from_rabi_mhz(
        cls,
        rabi_mhz: Any,
        dt_ns: float,
        *,
        channels: Sequence[str] | None = ("I", "Q"),
        **kwargs: Any,
    ) -> Pulse:
        """Build a ``'rad/ns'`` pulse from Rabi frequencies in MHz (shape ``(n, channels)``).

        Inverse of :meth:`rabi_mhz` up to rounding. ``kwargs`` go to :meth:`from_actions`
        (``task_id``, ``task_kwargs``, metadata); ``units`` is fixed to ``'rad/ns'``.
        """
        if kwargs.get("units", UNITS_RAD_PER_NS) != UNITS_RAD_PER_NS:
            msg = "from_rabi_mhz always produces 'rad/ns' pulses"
            raise ValueError(msg)
        kwargs["units"] = UNITS_RAD_PER_NS
        return cls.from_actions(mhz_to_rad_per_ns(rabi_mhz), dt_ns, channels=channels, **kwargs)

    # ------------------------------------------------------------- accessors
    def to_actions(self) -> np.ndarray:
        """Writable float64 copy of the samples, shape ``(n, channels)``: ``env.rollout`` input."""
        return np.array(self._samples)

    def rabi_mhz(self) -> np.ndarray:
        """Rabi frequency in MHz of every sample, ``samples / (2 pi) * 1e3`` (``'rad/ns'`` only)."""
        if self._units != UNITS_RAD_PER_NS:
            msg = (
                f"MHz conversion needs units {UNITS_RAD_PER_NS!r} (angular frequency in rad/ns), "
                f"this pulse has units {self._units!r}"
            )
            raise ValueError(msg)
        return rad_per_ns_to_mhz(self._samples)

    def area(self) -> np.ndarray:
        """Per-channel area ``sum(samples) * dt_ns``, shape ``(channels,)`` (rad for rad/ns).

        On resonance a constant-phase drive with area ``theta`` on channel I rotates a
        two-level system by ``theta`` about x (pi for an X gate).
        """
        return self._samples.sum(axis=0) * self._dt_ns

    # ------------------------------------------------------------ transforms
    def resample(self, sample_dt_ns: float) -> Pulse:
        """Refine the pulse onto a finer grid by sample-and-hold; exact for piecewise-constant.

        Every sample is repeated ``k = dt_ns / sample_dt_ns`` times and the new period is
        ``dt_ns / k``. ``k`` must be an integer to within ``1e-9`` relative; a ratio below 1
        (coarsening) or a non-integer ratio raises ``ValueError``, because the held pulse
        would no longer be the same function of time. The metadata records the factor in
        ``metadata['resampled']``; a ``task_kwargs`` entry still describes the original task.
        An existing ``metadata['unpadded_n_samples']`` is multiplied by ``k`` so that it keeps
        counting samples of the new grid; the intended order is resample, then pad.
        """
        s = float(sample_dt_ns)
        if not (math.isfinite(s) and s > 0):
            msg = f"sample_dt_ns must be positive and finite, got {sample_dt_ns!r}"
            raise ValueError(msg)
        ratio = self._dt_ns / s
        k = round(ratio)
        if k < 1 or abs(ratio - k) > RESAMPLE_RTOL * k:
            msg = (
                f"cannot resample a pulse with dt_ns = {self._dt_ns!r} to sample_dt_ns = {s!r}: "
                f"the ratio dt_ns / sample_dt_ns = {ratio!r} is not a positive integer (to "
                f"{RESAMPLE_RTOL:g} relative). Sample-and-hold is exact only when each control "
                "interval holds a whole number of hardware samples, e.g. 2 ns on a 2/9 ns grid "
                "(ratio 9); a coarser or incommensurate grid would change the pulse"
            )
            raise ValueError(msg)
        if k == 1:
            return self
        meta = self.metadata
        prev = meta.get("resampled")
        meta["resampled"] = (
            {"factor": prev["factor"] * k, "from_dt_ns": prev["from_dt_ns"]}
            if isinstance(prev, dict) and "factor" in prev and "from_dt_ns" in prev
            else {"factor": k, "from_dt_ns": self._dt_ns}
        )
        if isinstance(meta.get("unpadded_n_samples"), int):
            meta["unpadded_n_samples"] *= k
        return self._replace(np.repeat(self._samples, k, axis=0), self._dt_ns / k, meta)

    def pad(self, n_samples: int | None = None, *, multiple: int | None = None) -> Pulse:
        """Append zeros: to a total of ``n_samples`` samples, or to the next ``multiple``.

        Give exactly one of the two. ``pad(multiple=16)`` meets an AWG granularity of 16
        samples. The padded pulse records the original length in
        ``metadata['unpadded_n_samples']``; :meth:`resample` scales it, so resample first
        and pad second. The zeros are idle time: when a padded pulse is simulated, the drift
        (detuning, anharmonicity) and, in an open system, T1/T2 decay act during it, whereas
        hardware frame tracking does not.
        """
        if (n_samples is None) == (multiple is None):
            msg = "give exactly one of n_samples (total length) and multiple (granularity)"
            raise ValueError(msg)
        n = self.n_samples
        if multiple is not None:
            if int(multiple) != multiple or multiple < 1:
                msg = f"multiple must be a positive integer, got {multiple!r}"
                raise ValueError(msg)
            target = -(-n // int(multiple)) * int(multiple)
        else:
            if int(n_samples) != n_samples:  # type: ignore[arg-type]
                msg = f"n_samples must be an integer, got {n_samples!r}"
                raise ValueError(msg)
            target = int(n_samples)  # type: ignore[arg-type]
            if target < n:
                msg = (
                    f"cannot pad to {target} samples: the pulse already has {n}. "
                    "For AWG granularity use pad(multiple=N)"
                )
                raise ValueError(msg)
        if target == n:
            return self
        meta = self.metadata
        meta.setdefault("unpadded_n_samples", n)
        out = np.zeros((target, self.n_channels))
        out[:n] = self._samples
        return self._replace(out, None, meta)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dictionary with ``'schema': 'qenvs.pulse/v1'``."""
        return {
            "schema": SCHEMA,
            "dt_ns": self._dt_ns,
            "channels": list(self._channels),
            "units": self._units,
            "samples": self._samples.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pulse:
        """Inverse of :meth:`to_dict`; rejects unknown schemas and missing fields."""
        if not isinstance(data, Mapping):
            msg = f"expected a mapping, got {type(data).__name__}"
            raise TypeError(msg)
        schema = data.get("schema")
        if schema != SCHEMA:
            msg = f"unsupported pulse schema {schema!r}; this version reads {SCHEMA!r}"
            raise ValueError(msg)
        missing = [k for k in ("dt_ns", "channels", "units", "samples") if k not in data]
        if missing:
            msg = f"pulse record is missing fields: {missing}"
            raise ValueError(msg)
        return cls(
            data["samples"], data["dt_ns"], data["channels"], data["units"], data.get("metadata")
        )

    def to_json(
        self, path: str | os.PathLike[str] | None = None, *, indent: int | None = None
    ) -> str:
        """JSON text, written to ``path`` too if given.

        The text is ASCII and floats use the shortest round-trip repr, so it is lossless.
        """
        text = json.dumps(self.to_dict(), allow_nan=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | bytes | os.PathLike[str] | Mapping[str, Any]) -> Pulse:
        """Read a pulse from JSON text, a path to a JSON file or an already parsed mapping.

        A ``str`` whose first non-blank character is ``{`` is JSON text; any other ``str``
        and any path-like object is a file name.
        """
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if isinstance(source, str) and source.lstrip().startswith("{"):
            text = source
        else:
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def to_npz(self, path: PathOrFile, *, compressed: bool = False) -> None:
        """Write a NumPy ``.npz`` archive (no pickle); float64 samples are stored bit-exactly.

        A ``str`` or path is written as given (no ``.npz`` is appended behind your back);
        a binary file object is also accepted.
        """
        arrays = {
            "schema": np.array(SCHEMA),
            "samples": self._samples,
            "dt_ns": np.array(self._dt_ns),
            "channels": np.array(self._channels),
            "units": np.array(self._units),
            "metadata": np.array(self._meta),
        }
        save = np.savez_compressed if compressed else np.savez
        if isinstance(path, (str, os.PathLike)):
            with Path(path).open("wb") as f:
                save(f, **arrays)
        else:
            save(path, **arrays)

    @classmethod
    def from_npz(cls, path: PathOrFile) -> Pulse:
        """Read an archive written by :meth:`to_npz` (``allow_pickle=False``)."""
        with np.load(path, allow_pickle=False) as z:
            missing = [
                k for k in ("schema", "samples", "dt_ns", "channels", "units") if k not in z.files
            ]
            if missing:
                msg = f"not a qenvs pulse archive: missing arrays {missing}"
                raise ValueError(msg)
            data = {
                "schema": str(z["schema"]),
                "samples": z["samples"],
                "dt_ns": float(z["dt_ns"]),
                "channels": [str(c) for c in z["channels"]],
                "units": str(z["units"]),
                "metadata": json.loads(str(z["metadata"])) if "metadata" in z.files else {},
            }
        return cls.from_dict(data)


# ------------------------------------------------------------ device properties
@dataclass(frozen=True)
class DeviceProperties:
    """Device parameters in the units of :class:`~qenvs.models.transmon.Transmon`.

    ``None`` means "not given". ``frequency_ghz`` is not used by the rotating-frame model
    (only the detuning from the drive matters); keep it as pulse metadata.
    """

    frequency_ghz: float | None = None
    anharmonicity_ghz: float | None = None
    t1_us: float | None = None
    t2_us: float | None = None


#: quantity -> (explicit key with units, SI key of Qiskit BackendProperties, SI -> explicit factor)
_QUANTITIES: dict[str, tuple[str, str, float]] = {
    "frequency_ghz": ("frequency_ghz", "frequency", 1e-9),  # Hz -> GHz
    "anharmonicity_ghz": ("anharmonicity_ghz", "anharmonicity", 1e-9),  # Hz -> GHz
    "t1_us": ("t1_us", "T1", 1e6),  # s -> us
    "t2_us": ("t2_us", "T2", 1e6),  # s -> us
}
_KNOWN_KEYS = {k for pair in _QUANTITIES.values() for k in pair[:2]}
_UNIT_SUFFIXED = re.compile(r"^(t1|t2|frequency|anharmonicity)_", re.IGNORECASE)


def _number(key: str, value: Any) -> float:
    """Unwrap ``(value, timestamp)`` tuples and ``.value`` attributes into a finite float."""
    if isinstance(value, (tuple, list)):
        if not value:
            msg = f"property {key!r} is an empty sequence"
            raise ValueError(msg)
        value = value[0]
    elif hasattr(value, "value") and not isinstance(value, (int, float, np.number)):
        value = value.value
    if isinstance(value, bool):
        msg = f"property {key!r} must be a number, got {value!r}"
        raise TypeError(msg)
    try:
        x = float(value)
    except (TypeError, ValueError) as e:
        msg = f"property {key!r} must be a number or (number, timestamp), got {value!r}"
        raise TypeError(msg) from e
    if not math.isfinite(x):
        msg = f"property {key!r} must be finite, got {x}"
        raise ValueError(msg)
    return x


def parse_properties(props: Mapping[str, Any]) -> DeviceProperties:
    """Read device parameters from a mapping, with units in the key names or in SI.

    Accepted keys (other keys, such as readout errors, are ignored):

    * explicit units: ``frequency_ghz``, ``anharmonicity_ghz``, ``t1_us``, ``t2_us``;
    * SI, as returned by Qiskit's ``BackendProperties.qubit_property(q)``, a mapping
      ``{'T1': (seconds, timestamp), 'T2': (seconds, timestamp), 'frequency': (Hz, timestamp),
      'anharmonicity': (Hz, timestamp)}``; every value may also be a bare number.

    Qiskit is not imported; anything shaped like this works. Supplying the same quantity in
    both forms, a key such as ``t1_ms`` (a unit-suffixed name this function does not
    know), ``T2 > 2 T1``, or a non-positive T1 or T2 raises an error instead of guessing.
    """
    if not isinstance(props, Mapping):
        msg = f"props must be a mapping of property names to values, got {type(props).__name__}"
        raise TypeError(msg)
    for key in props:
        if isinstance(key, str) and key not in _KNOWN_KEYS and _UNIT_SUFFIXED.match(key):
            msg = (
                f"unsupported property name {key!r}: use frequency_ghz, anharmonicity_ghz, "
                "t1_us, t2_us (explicit units) or T1, T2 [seconds], frequency, anharmonicity [Hz]"
            )
            raise ValueError(msg)
    found: dict[str, float | None] = {}
    for name, (explicit, si, factor) in _QUANTITIES.items():
        if explicit in props and si in props:
            msg = f"give {name} either as {explicit!r} or as {si!r}, not both"
            raise ValueError(msg)
        if explicit in props:
            found[name] = _number(explicit, props[explicit])
        elif si in props:
            found[name] = _number(si, props[si]) * factor
        else:
            found[name] = None
    for name in ("t1_us", "t2_us"):
        v = found[name]
        if v is not None and not v > 0:
            msg = f"{name} must be positive, got {v!r}"
            raise ValueError(msg)
    t1, t2 = found["t1_us"], found["t2_us"]
    if t1 is not None and t2 is not None and t2 > 2.0 * t1 * (1.0 + 1e-12):
        msg = (
            f"T2 = {t2:g} us exceeds 2 T1 = {2.0 * t1:g} us, which is unphysical "
            "(1/T2 = 1/(2 T1) + 1/T_phi >= 1/(2 T1)). Measured T2 can exceed 2 T1 within "
            "its error bar; clip it yourself, for example t2_us = min(t2, 2 * t1), if that "
            "is what you intend"
        )
        raise ValueError(msg)
    return DeviceProperties(**found)
