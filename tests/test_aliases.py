from __future__ import annotations

import pytest

from converge_h5_reader import DEFAULT_ALIASES, AliasRegistry, ConvergeFile


@pytest.mark.parametrize(
    ("alias", "dataset"),
    [
        ("T", "TEMPERATURE"),
        ("P", "PRESSURE"),
        ("rho", "DENSITY"),
        ("u", "VELOCITY_X"),
        ("Y_H2", "MASSFRAC_H2"),
        ("X_O2", "MOLEFRAC_O2"),
        ("omega_H2", "MASS_SOURCE_H2"),
    ],
)
def test_resolve(alias, dataset):
    assert DEFAULT_ALIASES.resolve(alias) == dataset


def test_unknown_names_pass_through():
    assert DEFAULT_ALIASES.resolve("WEIRD_FIELD") == "WEIRD_FIELD"
    # A bare prefix is not a species name.
    assert DEFAULT_ALIASES.resolve("Y_") == "Y_"


def test_alias_for_is_the_inverse():
    assert DEFAULT_ALIASES.alias_for("TEMPERATURE") == "T"
    assert DEFAULT_ALIASES.alias_for("MASSFRAC_H2") == "Y_H2"
    assert DEFAULT_ALIASES.alias_for("UNMAPPED") == "UNMAPPED"


def test_with_does_not_mutate_the_default():
    extended = DEFAULT_ALIASES.with_(HRR="MY_HEAT_RELEASE")
    assert extended.resolve("HRR") == "MY_HEAT_RELEASE"
    assert DEFAULT_ALIASES.resolve("HRR") == "HEAT_GEN"


def test_custom_registry_is_honoured_end_to_end(synthetic_h5):
    aliases = AliasRegistry({"temp": "TEMPERATURE"})
    with ConvergeFile(synthetic_h5, aliases=aliases) as f:
        assert f[0].cells["temp"][0] == pytest.approx(300.0)
