"""Deterministic, integer-only randomness.

Instance generators must produce byte-identical instances on every platform,
Python build, and CPU forever.  ``random`` is not a safe basis for that: its
stream is a CPython implementation detail, and anything routed through libm
(``gauss``, ``expovariate``, ``lognormvariate``) can differ in the last bit
between glibc, musl, and macOS.  A single flipped bit changes an instance and
silently invalidates every published reference cost.

So this module is closed over the integers.  The core is splitmix64 -- five
lines of 64-bit arithmetic with no lookup tables and no floating point.  Every
derived distribution below is built from ``u64`` using integer operations only.
``math.isqrt`` is the one non-trivial function used anywhere in generation, and
it is exact by definition.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, TypeVar

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

T = TypeVar("T")


def fnv1a64(data: "str | bytes") -> int:
    """FNV-1a. Used to turn labels into seeds; stable across everything."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & MASK64
    return h


class Rng:
    """splitmix64. Cheap to seed, cheap to split, identical everywhere."""

    __slots__ = ("_s",)

    def __init__(self, seed: "int | str") -> None:
        if isinstance(seed, str):
            seed = fnv1a64(seed)
        self._s = seed & MASK64

    # -- core ---------------------------------------------------------------

    def u64(self) -> int:
        self._s = (self._s + _GOLDEN) & MASK64
        z = self._s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n: int) -> int:
        """Uniform in [0, n). Rejection sampled, so exactly uniform."""
        if n <= 0:
            raise ValueError(f"below() needs a positive bound, got {n}")
        if n & (n - 1) == 0:  # power of two: no rejection needed
            return self.u64() & (n - 1)
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = self.u64()
            if value < limit:
                return value % n

    def randint(self, low: int, high: int) -> int:
        """Uniform in [low, high], inclusive on both ends."""
        if high < low:
            raise ValueError(f"empty range [{low}, {high}]")
        return low + self.below(high - low + 1)

    # -- derived ------------------------------------------------------------

    def chance(self, numerator: int, denominator: int) -> bool:
        """True with probability numerator/denominator, exactly."""
        return self.below(denominator) < numerator

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.below(len(seq))]

    def weighted_choice(self, weights: Sequence[int]) -> int:
        """Index chosen proportionally to integer weights."""
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        pick = self.below(total)
        upto = 0
        for index, weight in enumerate(weights):
            upto += weight
            if pick < upto:
                return index
        return len(weights) - 1  # unreachable for positive weights

    def shuffle(self, items: List[T]) -> None:
        """In-place Fisher-Yates."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, population: Sequence[T], k: int) -> List[T]:
        """k distinct items, order randomised. Partial shuffle, O(k)."""
        if k > len(population):
            raise ValueError(f"cannot sample {k} from {len(population)}")
        pool = list(population)
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def normal(self, mean: int, sigma: int) -> int:
        """Irwin-Hall approximation of a normal, in pure integer arithmetic.

        Twelve uniforms summed and re-centred has variance 1 exactly, which is
        why this particular constant shows up in every generator that wants a
        bell without touching libm. Tails are clipped at +/- 6 sigma, which is
        a feature: instance coordinates stay bounded.
        """
        total = 0
        for _ in range(12):
            total += self.below(1 << 16)
        centred = total - 6 * (1 << 16)
        return mean + (sigma * centred >> 16)

    def skewed(self, low: int, high: int, power: int) -> int:
        """Integer draw from [low, high] biased toward ``low``.

        Takes the minimum of ``power`` uniform draws, which is the discrete
        analogue of a Beta(1, power): cheap, exact, and monotone in ``power``.
        Heavy-tailed families invert this to bias toward ``high``.
        """
        if power < 1:
            raise ValueError("power must be >= 1")
        best = self.randint(low, high)
        for _ in range(power - 1):
            candidate = self.randint(low, high)
            if candidate < best:
                best = candidate
        return best

    def spawn(self, label: str) -> "Rng":
        """A child stream. Independent, and reproducible from the label.

        Generators use this so that adding a new randomised feature does not
        shift every draw that came after it -- each concern owns its stream.
        """
        return Rng((self.u64() ^ fnv1a64(label)) & MASK64)


def derive_seed(*parts: "int | str") -> int:
    """Combine labels and numbers into one stable 64-bit seed."""
    h = 0xCBF29CE484222325
    for part in parts:
        h = (h ^ fnv1a64(str(part))) & MASK64
        h = (h * 0x100000001B3) & MASK64
    return h


def digest_ints(values: Iterable[int]) -> int:
    """Order-sensitive 64-bit digest of an integer stream.

    Used to fingerprint instances (so a solver mutating one is detectable) and
    candidate streams (so determinism can be checked by replay).
    """
    h = 0xCBF29CE484222325
    for value in values:
        value &= MASK64
        h = ((h ^ value) * 0x100000001B3) & MASK64
        h ^= h >> 29
    return h
