"""Offset distributions for parameter randomisation and control errors.

Every distribution draws an *offset* ``delta`` from the nominal value (the parameter value
is ``nominal + delta``), in the unit of the parameter. All three are frozen dataclasses with
a ``unit`` field (empty means "not stated"; a stated unit must equal the unit declared by
the model), analytic ``mean`` and ``stddev``, and ``sample(key)``.

* ``Normal(std)``: scales ``std`` with ``strength``; allowed only if the support of the
  parameter is the whole real line.
* ``Uniform(half_width)``: scales ``half_width``; ``[nominal - h, nominal + h]`` with the
  effective half-width ``h`` must lie inside the support.
* ``TruncatedNormal(std, low, high)``: scales ``std`` only (``low`` and ``high`` are fixed
  offsets); ``low <= 0 <= high``, ``low < high`` and ``[nominal + low, nominal + high]``
  must lie inside the support.

Order of operations (fixed): (1) the configured width is multiplied by ``strength`` to give
the effective width, (2) the range of values the effective distribution can produce is
checked against the support of the parameter at construction of the wrapper, (3) samples are
drawn. Samples are never clipped to the support, since that would change the declared
distribution; the only operation after the draw is a guard against floating-point rounding
(``TruncatedNormal`` and ``Uniform`` offsets are kept inside their own interval). A zero
effective width gives an offset of exactly ``0`` (also for ``TruncatedNormal``).

``TruncatedNormal.low`` and ``high`` are offsets from the nominal value and do not grow with
``strength``; when they are not symmetric the mean offset is not zero, and :meth:`mean` and
:meth:`stddev` give the exact values.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["Distribution", "Normal", "TruncatedNormal", "Uniform", "check_strength"]

_INF = math.inf
_SQRT2 = math.sqrt(2.0)
_NARROW = 0.05
"""Standardised interval length below which the truncated-normal moments use quadrature."""


def check_strength(strength: float) -> float:
    """Return ``strength`` as a float; raise ``ValueError`` unless it is finite and >= 0."""
    try:
        s = float(strength)
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError) as e:
        msg = f"strength must be a concrete finite number >= 0, got {strength!r}"
        raise ValueError(msg) from e
    if not (math.isfinite(s) and s >= 0.0):
        msg = f"strength must be a finite number >= 0, got {strength!r}"
        raise ValueError(msg)
    return s


def _finite_nonneg(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        msg = f"{name} must be a finite number >= 0, got {value!r}"
        raise ValueError(msg)
    return v


def _dtype(dtype: object | None) -> jnp.dtype:
    return jnp.result_type(float) if dtype is None else jnp.dtype(dtype)


def _phi(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    """Standard normal CDF (via ``erfc``, accurate in both tails)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _truncnorm_std_moments(a: float, b: float) -> tuple[float, float]:
    """Mean and standard deviation of ``N(0, 1)`` truncated to ``[a, b]`` (``a < b``).

    Closed form ``mean = (phi(a) - phi(b)) / Z`` and
    ``var = 1 + (a phi(a) - b phi(b)) / Z - mean^2`` with ``Z = Phi(b) - Phi(a)`` (callers
    guarantee ``a <= 0 <= b``, so ``Z`` never underflows); for narrow intervals, where those
    expressions cancel, Gauss-Legendre quadrature.
    """
    if b - a < _NARROW:
        x, w = np.polynomial.legendre.leggauss(64)
        x = 0.5 * (b - a) * x + 0.5 * (b + a)
        w = 0.5 * (b - a) * w * np.exp(-0.5 * x * x)
        z = w.sum()
        mean = float((w * x).sum() / z)
        var = float((w * (x - mean) ** 2).sum() / z)
        return mean, math.sqrt(max(var, 0.0))
    z = _cdf(b) - _cdf(a)
    mean = (_phi(a) - _phi(b)) / z
    var = 1.0 + (a * _phi(a) - b * _phi(b)) / z - mean * mean
    return mean, math.sqrt(max(var, 0.0))


class Distribution(abc.ABC):
    """Offset distribution (base class of ``Normal``, ``Uniform`` and ``TruncatedNormal``).

    Subclasses are frozen dataclasses. All methods take ``strength``, the factor that
    turns the configured width into the effective width (see the module docstring).
    """

    unit: str

    @abc.abstractmethod
    def offset_range(self, strength: float = 1.0) -> tuple[float, float]:
        """Closed interval ``(lo, hi)`` of the offsets the effective distribution can give."""

    @abc.abstractmethod
    def mean(self, strength: float = 1.0) -> float:
        """Exact mean of the offset."""

    @abc.abstractmethod
    def stddev(self, strength: float = 1.0) -> float:
        """Exact standard deviation of the offset."""

    @abc.abstractmethod
    def sample(
        self,
        key: Array,
        strength: float | Array = 1.0,
        shape: tuple[int, ...] = (),
        dtype: object | None = None,
    ) -> Array:
        """Draw offsets of the given ``shape`` from ``key`` (works under ``jit`` and ``vmap``)."""

    def check_support(
        self,
        name: str,
        nominal: float,
        support: tuple[float, float],
        strength: float = 1.0,
    ) -> None:
        """Raise ``ValueError`` if the effective distribution can leave ``support``.

        ``nominal`` is the nominal parameter value and ``support`` the closed interval
        declared by the model. Only the range of attainable values is checked; nothing is
        ever clipped.
        """
        strength = check_strength(strength)
        lo, hi = self.offset_range(strength)
        if nominal + lo < support[0] or nominal + hi > support[1]:
            msg = (
                f"{name}: {type(self).__name__} can produce values in "
                f"[{nominal + lo:g}, {nominal + hi:g}] (nominal {nominal:g}, strength "
                f"{strength:g}) but the support of the parameter is [{support[0]:g}, "
                f"{support[1]:g}]; reduce the width or the strength, or use a "
                "TruncatedNormal whose low and high stay inside the support"
            )
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian offset ``N(0, (std * strength)^2)``.

    Allowed only for parameters whose support is the whole real line (a Gaussian has
    unbounded tails, so it cannot honour a bounded support); use :class:`Uniform` or
    :class:`TruncatedNormal` otherwise.

    Attributes:
        std: standard deviation in the unit of the parameter (``>= 0``).
        unit: unit of ``std``; empty means "not stated".
    """

    std: float
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "std", _finite_nonneg("std", self.std))

    def offset_range(self, strength: float = 1.0) -> tuple[float, float]:  # noqa: ARG002
        return (-_INF, _INF)

    def mean(self, strength: float = 1.0) -> float:  # noqa: ARG002
        return 0.0

    def stddev(self, strength: float = 1.0) -> float:
        return self.std * check_strength(strength)

    def check_support(
        self,
        name: str,
        nominal: float,  # noqa: ARG002
        support: tuple[float, float],
        strength: float = 1.0,
    ) -> None:
        check_strength(strength)
        if not (math.isinf(support[0]) and math.isinf(support[1])):
            msg = (
                f"{name}: Normal needs a parameter whose support is the whole real line, but "
                f"the support is [{support[0]:g}, {support[1]:g}]; a Gaussian has unbounded "
                "tails. Use Uniform or TruncatedNormal(std, low, high)"
            )
            raise ValueError(msg)

    def sample(
        self,
        key: Array,
        strength: float | Array = 1.0,
        shape: tuple[int, ...] = (),
        dtype: object | None = None,
    ) -> Array:
        dt = _dtype(dtype)
        sigma = jnp.asarray(self.std, dt) * jnp.asarray(strength, dt)
        z = jax.random.normal(key, shape, dt)
        return jnp.where(sigma > 0, sigma * z, jnp.zeros((), dt))


@dataclasses.dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform offset on ``[-h, h]`` with ``h = half_width * strength``.

    Attributes:
        half_width: configured half-width in the unit of the parameter (``>= 0``);
            the standard deviation is ``h / sqrt(3)``.
        unit: unit of ``half_width``; empty means "not stated".
    """

    half_width: float
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "half_width", _finite_nonneg("half_width", self.half_width))

    def offset_range(self, strength: float = 1.0) -> tuple[float, float]:
        h = self.half_width * check_strength(strength)
        return (-h, h)

    def mean(self, strength: float = 1.0) -> float:  # noqa: ARG002
        return 0.0

    def stddev(self, strength: float = 1.0) -> float:
        return self.half_width * check_strength(strength) / math.sqrt(3.0)

    def sample(
        self,
        key: Array,
        strength: float | Array = 1.0,
        shape: tuple[int, ...] = (),
        dtype: object | None = None,
    ) -> Array:
        dt = _dtype(dtype)
        h = jnp.asarray(self.half_width, dt) * jnp.asarray(strength, dt)
        u = jax.random.uniform(key, shape, dt, minval=-1.0, maxval=1.0)
        return jnp.where(h > 0, jnp.clip(h * u, -h, h), jnp.zeros((), dt))


@dataclasses.dataclass(frozen=True)
class TruncatedNormal(Distribution):
    """Gaussian offset truncated to ``[low, high]``: ``N(0, (std * strength)^2)`` on that interval.

    ``low`` and ``high`` are offsets from the nominal value (``low <= 0 <= high``,
    ``low < high``, both finite) and are *not* scaled by ``strength``; only ``std`` is.
    When ``low != -high`` the mean offset is not zero: :meth:`mean` and :meth:`stddev` are
    the exact moments of the truncated distribution.

    Attributes:
        std: standard deviation of the underlying Gaussian, in the unit of the parameter
            (``>= 0``; ``0`` gives an offset of exactly ``0``).
        low: lower end of the offset interval (``<= 0``).
        high: upper end of the offset interval (``>= 0``).
        unit: unit of ``std``, ``low`` and ``high``; empty means "not stated".
    """

    std: float
    low: float
    high: float
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "std", _finite_nonneg("std", self.std))
        low, high = float(self.low), float(self.high)
        if not (math.isfinite(low) and math.isfinite(high)):
            msg = f"low and high must be finite offsets, got low={self.low!r}, high={self.high!r}"
            raise ValueError(msg)
        if low > 0.0 or high < 0.0 or low >= high:
            msg = (
                "TruncatedNormal needs low <= 0 <= high and low < high (offsets from the "
                f"nominal value), got low={low:g}, high={high:g}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def offset_range(self, strength: float = 1.0) -> tuple[float, float]:
        check_strength(strength)
        return (self.low, self.high)

    def _moments(self, strength: float) -> tuple[float, float]:
        sigma = self.std * check_strength(strength)
        if sigma == 0.0:
            return 0.0, 0.0
        m, s = _truncnorm_std_moments(self.low / sigma, self.high / sigma)
        return sigma * m, sigma * s

    def mean(self, strength: float = 1.0) -> float:
        return self._moments(strength)[0]

    def stddev(self, strength: float = 1.0) -> float:
        return self._moments(strength)[1]

    def sample(
        self,
        key: Array,
        strength: float | Array = 1.0,
        shape: tuple[int, ...] = (),
        dtype: object | None = None,
    ) -> Array:
        dt = _dtype(dtype)
        sigma = jnp.asarray(self.std, dt) * jnp.asarray(strength, dt)
        live = sigma > 0
        safe = jnp.where(live, sigma, jnp.ones((), dt))
        lower = jnp.asarray(self.low, dt) / safe
        upper = jnp.asarray(self.high, dt) / safe
        z = jax.random.truncated_normal(key, lower, upper, shape, dt)
        offset = jnp.clip(safe * z, self.low, self.high)  # guard against rounding at the ends
        return jnp.where(live, offset, jnp.zeros((), dt))
