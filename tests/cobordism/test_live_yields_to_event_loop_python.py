# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""``--live`` must keep the window alive while the worker computes (#1008).

``drive_live`` runs the engine on a worker thread and draws on the main one,
and the engine releases the GIL, so the worker genuinely proceeds while the
main thread draws. What went wrong was how the main thread WAITED: a blocking
``queue.Queue.get()`` parked it for the whole of an engine unit, and for that
entire interval the GUI event loop was never serviced. The desktop reads a main
thread that has not drained its event queue as a hung application -- "this
application is not responding" -- and a unit can take minutes.

The loop now polls with ``get_nowait`` and, when nothing is ready, waits inside
``plt.pause``, which pumps the event loop and sleeps for the interval. The
observable difference is the number of times the event loop is serviced between
two finished units: once, before, and once per ``LIVE_POLL_INTERVAL`` now.

matplotlib is stubbed here rather than driven. The claim under test is about
the loop's waiting discipline, not about any backend's behaviour, and a real
GUI cannot be opened in a test process.
"""
import queue
import sys
import threading
import time

import pytest

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("/", 2)[0] + "/examples/cobordism")

import emergence_animation as ea  # noqa: E402

#: How long the fake worker spends "computing" one unit. Comfortably many poll
#: intervals, so the count below is not a race.
UNIT_SECONDS = 0.4


class FakeCanvas:
    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


class FakeFigure:
    def __init__(self):
        self.canvas = FakeCanvas()


def install_stubs(monkeypatch, pauses):
    """Enough of matplotlib for the loop to run, and nothing more."""
    import matplotlib
    import matplotlib.pyplot as plt

    monkeypatch.setattr(matplotlib, "get_backend", lambda: "qtagg")
    monkeypatch.setattr(plt, "isinteractive", lambda: True)
    monkeypatch.setattr(plt, "figure", lambda **kwargs: FakeFigure())
    monkeypatch.setattr(plt, "close", lambda figure: None)

    def pause(interval):
        pauses.append(interval)
        time.sleep(min(interval, 0.01))

    monkeypatch.setattr(plt, "pause", pause)
    monkeypatch.setattr(ea, "draw_frame", lambda *a, **k: None)
    monkeypatch.setattr(ea, "place_frame", lambda state, frame: None)


def slow_drive(units):
    """A stand-in for `drive` that takes real time per unit, as the engine does."""
    def driver(config, progress=False, on_frame=None, on_node=None,
               on_setup=None, stop_requested=None):
        frames = []
        for step in range(units):
            time.sleep(UNIT_SECONDS)
            frames.append(object())
            if on_frame is not None:
                on_frame(frames, step)
        return ea.DriveResult([], ea.Terminator.STEPS, None, 0)
    return driver


def test_the_event_loop_is_serviced_while_the_worker_computes(monkeypatch):
    """The substance of the fix.

    One unit takes UNIT_SECONDS; at LIVE_POLL_INTERVAL the main thread should
    service the event loop many times over that span. Blocking in the queue --
    the old behaviour -- serviced it once per unit, which is what the desktop
    called a hang.
    """
    pauses = []
    install_stubs(monkeypatch, pauses)
    monkeypatch.setattr(ea, "drive", slow_drive(2))
    ea.drive_live(ea.build_config(size=4, steps=2), progress=False)
    expected = int(2 * UNIT_SECONDS / ea.LIVE_POLL_INTERVAL) // 2
    assert len(pauses) >= expected, (len(pauses), expected)
    assert all(interval == ea.LIVE_POLL_INTERVAL for interval in pauses)


def test_it_still_draws_every_unit_in_order_and_stops_on_the_sentinel(monkeypatch):
    """Polling must not change WHAT is drawn, only how the loop waits."""
    pauses = []
    install_stubs(monkeypatch, pauses)
    drawn = []
    monkeypatch.setattr(ea, "draw_frame",
                        lambda figure, frames, index, placed: drawn.append(index))
    monkeypatch.setattr(ea, "drive", slow_drive(3))
    result = ea.drive_live(ea.build_config(size=4, steps=3), progress=False)
    assert drawn == [0, 1, 2]
    assert result.terminator == ea.Terminator.STEPS


def test_a_worker_error_still_reaches_the_main_thread(monkeypatch):
    """The sentinel is posted in a finally, so a failing unit must not hang."""
    pauses = []
    install_stubs(monkeypatch, pauses)

    def exploding(config, progress=False, on_frame=None, on_node=None,
                  on_setup=None, stop_requested=None):
        time.sleep(UNIT_SECONDS / 4)
        raise ValueError("the unit failed")

    monkeypatch.setattr(ea, "drive", exploding)
    try:
        ea.drive_live(ea.build_config(size=4, steps=1), progress=False)
    except ValueError as error:
        assert "the unit failed" in str(error)
    else:
        raise AssertionError("the worker's error did not reach the main thread")


def test_the_loop_does_not_spin_when_the_queue_is_empty(monkeypatch):
    """Polling is not busy-waiting: each empty poll waits the interval.

    Asserted as an upper bound on how many times the event loop was serviced
    over a known span. A loop that called `get_nowait` without waiting would
    run away, and the count would be orders larger.
    """
    pauses = []
    install_stubs(monkeypatch, pauses)
    monkeypatch.setattr(ea, "drive", slow_drive(1))
    started = time.monotonic()
    ea.drive_live(ea.build_config(size=4, steps=1), progress=False)
    elapsed = time.monotonic() - started
    # The stubbed pause sleeps at most 10ms rather than the full interval, so
    # the ceiling is set by that, not by LIVE_POLL_INTERVAL.
    assert len(pauses) < (elapsed / 0.01) + 20, (len(pauses), elapsed)


def test_the_declared_interval_is_a_responsive_one():
    """A window serviced less than about ten times a second reads as sluggish."""
    assert 0.0 < ea.LIVE_POLL_INTERVAL <= 0.1


def test_interrupt_stops_and_joins_the_worker_before_returning(monkeypatch):
    """Interrupted geometry cannot race a still-mutating daemon worker."""
    pauses = []
    install_stubs(monkeypatch, pauses)
    active = threading.Event()
    ready = threading.Event()
    setup = {}

    def interruptible(config, progress=False, on_frame=None, on_node=None,
                      on_setup=None, stop_requested=None):
        active.set()
        if on_setup is not None:
            on_setup("node", "qubit-inputs")
        ready.set()
        while not stop_requested():
            time.sleep(0.005)
        active.clear()
        return ea.DriveResult([], ea.Terminator.STEPS, "qubit-inputs", 0)

    monkeypatch.setattr(ea, "drive", interruptible)
    import matplotlib.pyplot as plt

    def interrupt_after_setup(_interval):
        assert ready.wait(timeout=1.0), "worker never completed setup"
        raise KeyboardInterrupt

    monkeypatch.setattr(plt, "pause", interrupt_after_setup)
    with pytest.raises(KeyboardInterrupt):
        ea.drive_live(
            ea.build_config(size=4, steps=1), progress=False,
            on_setup=lambda node, inputs: setup.update(node=node, inputs=inputs))
    assert not active.is_set()
    assert setup == {"node": "node", "inputs": "qubit-inputs"}
