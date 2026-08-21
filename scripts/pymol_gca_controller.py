"""License-independent PyMOL controls for graph-CA trajectory objects."""

from __future__ import annotations

import time

from pymol import cmd


_STATE_COUNT = 18
_current_state = 1


def _trajectory_objects():
    return sorted(name for name in cmd.get_names("objects") if name.startswith("traj_"))


def gca_state(state=1):
    """Show and recolour one graph-CA state without using PyMOL movies."""
    global _current_state
    state = max(1, min(_STATE_COUNT, int(state)))
    _current_state = state
    cmd.set("state", state)
    objects = _trajectory_objects()
    for obj in objects:
        cmd.spectrum("b", "cyan_magenta", obj, minimum=0.0, maximum=100.0)
    hydrogen_selection = "(" + " or ".join(objects) + ") and elem H"
    if state == 18:
        cmd.show("sticks", hydrogen_selection)
        cmd.color("cyber_lime", hydrogen_selection)
    else:
        cmd.hide("sticks", hydrogen_selection)
    cmd.rebuild()
    cmd.refresh()
    print(f"Graph-CA display state {state}/18")


def gca_next():
    """Advance one state, wrapping from 18 to 1."""
    gca_state(1 if _current_state >= _STATE_COUNT else _current_state + 1)


def gca_previous():
    """Move back one state, wrapping from 1 to 18."""
    gca_state(_STATE_COUNT if _current_state <= 1 else _current_state - 1)


def gca_play(delay=0.45, cycles=1):
    """Play states 1-18 without invoking PyMOL's licensed movie subsystem."""
    delay = max(0.05, float(delay))
    cycles = max(1, int(cycles))
    for _ in range(cycles):
        for state in range(1, _STATE_COUNT + 1):
            gca_state(state)
            time.sleep(delay)


cmd.extend("gca_state", gca_state)
cmd.extend("gca_next", gca_next)
cmd.extend("gca_previous", gca_previous)
cmd.extend("gca_play", gca_play)

print("Graph-CA controls loaded: gca_next, gca_previous, gca_state 1-18, gca_play")

