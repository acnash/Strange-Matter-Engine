from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_graph_ca_visual_prototype import (
    ATOM_FEATURE_NAMES,
    ATOM_FEATURE_PROFILES,
    BASELINE_ATOM_FEATURES,
)


def test_atom_feature_names_are_unique():
    assert len(ATOM_FEATURE_NAMES) == len(set(ATOM_FEATURE_NAMES))


def test_every_profile_contains_the_existing_baseline():
    baseline = set(BASELINE_ATOM_FEATURES)
    assert all(baseline.issubset(profile)
               for profile in map(set, ATOM_FEATURE_PROFILES.values()))


def test_comprehensive_profile_contains_every_available_property():
    assert tuple(ATOM_FEATURE_PROFILES["comprehensive"]) == ATOM_FEATURE_NAMES


def test_search_includes_single_group_and_combination_profiles():
    assert {"periodic", "valence", "electronic", "ring_geometry"}.issubset(
        ATOM_FEATURE_PROFILES
    )
    assert {"periodic_valence", "periodic_electronic",
            "valence_electronic", "comprehensive"}.issubset(ATOM_FEATURE_PROFILES)
