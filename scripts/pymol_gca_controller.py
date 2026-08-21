"""License-independent PyMOL controls for graph-CA trajectory objects."""

import time
import threading

from pymol import cmd


_STATE_COUNT = int(globals().get("GCA_STATE_COUNT", 18))
_current_state = 1
_play_thread = None
_stop_playback = threading.Event()
GCA_DISPLAY_VALUES = globals().get("GCA_DISPLAY_VALUES", {})


def _trajectory_objects():
    return sorted(name for name in cmd.get_names("objects", enabled_only=1)
                  if name.startswith("traj_"))


def gca_state(state=1):
    """Show and recolour one graph-CA state without using PyMOL movies."""
    global _current_state
    state = max(1, min(_STATE_COUNT, int(state)))
    _current_state = state
    cmd.set("state", state)
    objects = _trajectory_objects()
    for obj in objects:
        if obj in GCA_DISPLAY_VALUES:
            values = iter(GCA_DISPLAY_VALUES[obj][state - 1])
            cmd.alter(obj, "b=next(gca_values)", space={"gca_values": values})
        cmd.spectrum("b", "cyan_magenta", obj, minimum=0.0, maximum=100.0)
    if objects:
        hydrogen_selection = "(" + " or ".join(objects) + ") and elem H"
        if state == 18:
            cmd.show("sticks", hydrogen_selection)
            cmd.color("cyber_lime", hydrogen_selection)
        else:
            cmd.hide("sticks", hydrogen_selection)
    cmd.refresh()
    print(f"Graph-CA display state {state}/{_STATE_COUNT}")


def gca_next():
    """Advance one state, wrapping from 18 to 1."""
    gca_state(1 if _current_state >= _STATE_COUNT else _current_state + 1)


def gca_previous():
    """Move back one state, wrapping from 1 to 18."""
    gca_state(_STATE_COUNT if _current_state <= 1 else _current_state - 1)


def _play_worker(delay, cycles):
    for _ in range(cycles):
        for state in range(1, _STATE_COUNT + 1):
            if _stop_playback.is_set():
                return
            gca_state(state)
            time.sleep(delay)


def gca_play(delay=0.45, cycles=1):
    """Play asynchronously without invoking PyMOL's movie subsystem."""
    global _play_thread
    delay = max(0.05, float(delay))
    cycles = max(1, int(cycles))
    if _play_thread is not None and _play_thread.is_alive():
        print("Graph-CA playback is already running; use gca_stop first")
        return
    _stop_playback.clear()
    _play_thread = threading.Thread(target=_play_worker, args=(delay, cycles), daemon=True)
    _play_thread.start()
    print(f"Graph-CA playback started: {cycles} cycle(s), {delay:.2f} s per state")


def gca_stop():
    """Stop asynchronous graph-CA playback."""
    _stop_playback.set()
    print("Graph-CA playback stop requested")


cmd.extend("gca_state", gca_state)
cmd.extend("gca_next", gca_next)
cmd.extend("gca_previous", gca_previous)
cmd.extend("gca_play", gca_play)
cmd.extend("gca_stop", gca_stop)

print(f"Graph-CA controls loaded: gca_next, gca_previous, gca_state 1-{_STATE_COUNT}, gca_play, gca_stop")
