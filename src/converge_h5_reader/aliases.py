"""Short-name to dataset-name resolution for CONVERGE cell variables.

CONVERGE stores cell variables under verbose names (``TEMPERATURE``,
``MASSFRAC_H2``).  An :class:`AliasRegistry` maps convenient short names onto
them so callers can write ``cells["T"]`` or ``cells["Y_H2"]``.

The registry is a resolution *rule set*, not a required field list: unknown
names pass through untouched and only fail (with :class:`MissingFieldError`)
if the underlying dataset really is absent.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_ALIASES", "DEFAULT_PREFIXES", "AliasRegistry"]

#: Prefix rules applied when a name is not in the literal mapping.
#: ``Y_H2`` -> ``MASSFRAC_H2``, ``X_O2`` -> ``MOLEFRAC_O2``, ...
DEFAULT_PREFIXES: Mapping[str, str] = {
    "Y_": "MASSFRAC_",
    "X_": "MOLEFRAC_",
    "omega_": "MASS_SOURCE_",
    "Vd_": "SPECIES_DIFF_VEL_",
}

_DEFAULT_MAPPING: Mapping[str, str] = {
    "x": "XCEN_X",
    "y": "XCEN_Y",
    "z": "XCEN_Z",
    "u": "VELOCITY_X",
    "v": "VELOCITY_Y",
    "w": "VELOCITY_Z",
    "T": "TEMPERATURE",
    "P": "PRESSURE",
    "rho": "DENSITY",
    "h": "ENTHALPY",
    "phi": "EQUIV_RATIO",
    "mu": "MOL_VISC",
    "mu_t": "TURB_VISCOSITY",
    "V": "VOLUME",
    "m": "MASS",
    "HRR": "HEAT_GEN",
    "WHF": "BOUND_FLUX",
    "region": "REGION_ID",
    "wx": "VORTICITY_X",
    "wy": "VORTICITY_Y",
    "wz": "VORTICITY_Z",
    "y_plus": "YPLUS",
    "wall_dist": "WALL_DIST",
}


@dataclass(frozen=True)
class AliasRegistry:
    """Immutable, composable alias -> dataset mapping."""

    mapping: Mapping[str, str] = field(default_factory=dict)
    prefixes: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_PREFIXES))

    def resolve(self, name: str) -> str:
        """Return the dataset name for ``name``, or ``name`` itself if it is not an alias."""
        if name in self.mapping:
            return self.mapping[name]
        for short, full in self.prefixes.items():
            if name.startswith(short) and len(name) > len(short):
                return full + name[len(short) :]
        return name

    def alias_for(self, dataset: str) -> str:
        """Return the short name for a dataset, or the dataset name if none is registered."""
        for alias, target in self.mapping.items():
            if target == dataset:
                return alias
        for short, full in self.prefixes.items():
            if dataset.startswith(full) and len(dataset) > len(full):
                return short + dataset[len(full) :]
        return dataset

    def with_(self, **extra: str) -> AliasRegistry:
        """Return a copy with additional alias -> dataset entries."""
        return self.updated(extra)

    def updated(self, other: Mapping[str, str]) -> AliasRegistry:
        """Return a copy with ``other`` merged over the current mapping."""
        return AliasRegistry({**self.mapping, **other}, dict(self.prefixes))

    def with_prefixes(self, **extra: str) -> AliasRegistry:
        """Return a copy with additional prefix rules."""
        return AliasRegistry(dict(self.mapping), {**self.prefixes, **extra})

    @classmethod
    def from_toml(cls, path: str | Path) -> AliasRegistry:
        """Load a registry from TOML with optional ``[aliases]`` and ``[prefixes]`` tables."""
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - exercised only on Python < 3.11
            import tomli as tomllib

        with Path(path).open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        return cls(
            dict(data.get("aliases", {})),
            dict(data.get("prefixes", DEFAULT_PREFIXES)),
        )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.resolve(name) != name

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)


DEFAULT_ALIASES = AliasRegistry(dict(_DEFAULT_MAPPING), dict(DEFAULT_PREFIXES))
