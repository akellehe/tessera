# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The animation an example draws, independent of what the example computes.

Every cobordism example that animates does the same four things: it keeps a
frame's drawing STABLE as the complex changes under it, it lays a set of
panels into a figure, it drives the compute on a worker thread while the main
thread draws, and it renders the finished frames to a file. None of that
depends on what is being animated.

What DOES depend on the experiment is the panels, the title, and the driver,
so each of those is passed in. An example supplies a list of panel callables,
a title function and its own `drive`, and gets the live window and the
renderer for free.

The split is where the mode-neutrality already was: these six pieces were
shared verbatim between the neutral and qubit drivers before they lived here,
and a third example (the level-0 velocity run) needed them without wanting
either driver's panels.
"""
import math
import os
import queue
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#: How long the live loop sleeps between polls of the worker's frame queue.
#: `plt.pause` both pumps the GUI event loop and sleeps for the interval, so
#: this is the redraw cadence and the poll cadence at once.
LIVE_POLL_INTERVAL = 0.05

#: Stabilization of the drawing layout. Classical MDS is defined only up to
#: rotation, reflection and scale and is globally sensitive, so a small change
#: to the complex reshuffles the whole cloud. Scale is already fixed by the
#: RMS normalization the layout read performs; these three remove the
#: orientation ambiguity, ease the positions, and ease the view.
DECLARED_LAYOUT_EASE = 0.3
DECLARED_LAYOUT_VIEW_EASE = 0.25
DECLARED_LAYOUT_PAD = 0.18
#: Grid the panels are laid out on. Wide enough for every panel with room to
#: spare; the spare axes are removed rather than left as empty boxes, which
#: would read as absent measurements.
DECLARED_PANEL_GRID = (4, 5)
#: The live window and the rendered canvas, in inches, sized for the 4x5 grid
#: above. Both are overridable per example, alongside `grid` on `_draw_frame`,
#: so panel count and canvas stay in proportion.
DECLARED_LIVE_FIGSIZE = (18, 10)
DECLARED_RENDER_FIGSIZE = (20, 12)


class StableLayout:
    """Jitter-free drawing positions: the layout read's normalized embedding,
    rigidly aligned to the previous frame and eased toward it, with an eased
    auto-fit view.

    Classical MDS is defined only up to rotation, reflection and scale, and it
    is globally sensitive, so a small change to the complex can reshuffle the
    whole cloud. Scale is already fixed by the read's RMS normalization; this
    removes the orientation ambiguity by Procrustes, eases the positions, and
    auto-fits the view, so the structure stays legible as the complex changes.

    PRESENTATION ONLY. It consumes what the layout read measured and produces
    where to draw it; it feeds nothing back into any measurement or record.
    """

    def __init__(self, ease=DECLARED_LAYOUT_EASE,
                 view_ease=DECLARED_LAYOUT_VIEW_EASE,
                 pad=DECLARED_LAYOUT_PAD):
        self._previous = None
        self._view = None
        self.ease = ease
        self.view_ease = view_ease
        self.pad = pad

    def place(self, coords):
        """Stabilized positions for one frame's raw layout coordinates."""
        import numpy as np

        current = {v: np.asarray(p, dtype=float) for v, p in coords.items()}
        if len(current) < 2 or self._previous is None:
            # The first frame defines the frame of reference; there is nothing
            # to align to and nothing to ease toward.
            self._previous = current
            return {v: tuple(p) for v, p in current.items()}
        shared = [v for v in current if v in self._previous]
        if len(shared) >= 2:
            cur = np.array([current[v] for v in shared])
            ref = np.array([self._previous[v] for v in shared])
            cur_centre, ref_centre = cur.mean(0), ref.mean(0)
            u, _s, vt = np.linalg.svd((cur - cur_centre).T
                                      @ (ref - ref_centre))
            rotation = u @ vt          # rotation/reflection only, no scale
            aligned = {v: (p - cur_centre) @ rotation + ref_centre
                       for v, p in current.items()}
        else:
            # Too little in common to define an alignment. Taking the raw
            # embedding is honest; inventing a rotation from one point is not.
            aligned = current
        eased = {}
        for vertex, target in aligned.items():
            previous = self._previous.get(vertex)
            # A vertex that has just appeared has nowhere to ease FROM, so it
            # takes its target outright rather than sliding in from a position
            # it never occupied.
            eased[vertex] = (target if previous is None
                             else previous + self.ease * (target - previous))
        self._previous = eased
        return {v: tuple(p) for v, p in eased.items()}

    def view(self, coords):
        """An eased bounding box around the current cloud.

        Never grow-only: a view that could only expand would shrink the
        structure to an unreadable dot as soon as one frame spread out.
        """
        import numpy as np

        points = np.array(list(coords.values()), dtype=float)
        low, high = points.min(0), points.max(0)
        pad = self.pad * max(high[0] - low[0], high[1] - low[1], 1e-6)
        box = [low[0] - pad, high[0] + pad, low[1] - pad, high[1] + pad]
        if self._view is None:
            self._view = box
        else:
            self._view = [self._view[i]
                          + self.view_ease * (box[i] - self._view[i])
                          for i in range(4)]
        return self._view

def place_frame(state, frame):
    """One frame's stabilized placement, advancing `state` by one link.

    The single step both drivers share. `stabilize` walks it over a finished
    run and the live path calls it as each unit completes; because the
    alignment is a CHAIN, the two agree only if they feed the same state the
    same frames in the same order. Sharing the step makes that structural
    rather than a coincidence two call sites have to maintain.

    `None` where the layout itself is absent, so the chain skips a frame it
    cannot place rather than aligning the next one to nothing.
    """
    # An absent layout is recognised by the sentinel's defining attribute
    # rather than by its class: importing the driver's `Absent` here would be
    # a cycle, and an animation library has no business knowing what a
    # measurement channel is. A real layout carries no `reason`.
    if frame.layout is None or hasattr(frame.layout, "reason"):
        return None
    coords = state.place(frame.layout["coords"])
    return {"coords": coords, "view": state.view(coords)}

def draw_frame(figure, frames, index, panels, title_fn, trace_panels,
                placed_panels, placed=None, grid=None):
    """Draw one frame's panels onto a figure.

    `placed` is `stabilize(frames)`. Passing it is optional so a caller can
    draw a single frame without it, in which case the raw layout is used and
    the picture is correct but unaligned.

    `grid` is the (rows, columns) the panels are laid out on, defaulting to
    this module's own. An example with a different number of panels passes its
    own, so the grid tracks the panel list rather than every caller inheriting
    a grid sized for the emergence instrument.
    """
    figure.clear()
    frame = frames[index]
    placement = placed[index] if placed else None
    rows, columns = grid or DECLARED_PANEL_GRID
    axes = figure.subplots(rows, columns, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for axis in flat[len(panels):]:
        figure.delaxes(axis)
    for (name, painter), axis in zip(panels, flat):
        if name in trace_panels:
            painter(axis, frames[:index + 1])
        elif name in placed_panels:
            painter(axis, frame, placement)
        else:
            painter(axis, frame)
    figure.suptitle(title_fn(frame, frames[-1].step), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

def interactive_backends():
    """The interactive matplotlib backends, lowercased.

    Asked of matplotlib rather than hard-coded, so the set cannot drift out
    of step with the installed version. The fallback names the file-only
    backends instead, which is the smaller and far more stable list.
    """
    try:
        from matplotlib.backends import BackendFilter, backend_registry
        return {name.lower() for name in
                backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    except ImportError:                     # matplotlib < 3.9
        import matplotlib
        try:
            return {name.lower() for name in matplotlib.rcsetup.interactive_bk}
        except AttributeError:
            return set()

def drive_live(config, driver, drawer, progress=False, on_node=None,
                on_setup=None, thread_name="animation-drive", figsize=None):
    """Drive with supplied experiment callbacks while drawing live frames.

    The compute runs on a worker thread and the figure is drawn on the main
    one, because a GUI toolkit may only be driven from the thread that owns
    it. That is safe here rather than merely conventional: `run_stage1` and
    `run_stage2` release the GIL, so the worker genuinely proceeds while the
    main thread draws, and the worker only ever APPENDS to the frame list
    while the main thread reads indices it has already been handed.

    The supplied driver is unchanged and un-forked. A live run and a
    headless one execute the same loop with the same engine calls; only the
    callback differs. There is no second code path to diverge.
    """
    import queue
    import threading

    import matplotlib
    import matplotlib.pyplot as plt

    # Checked on the BACKEND, not on whether a figure can be created: a
    # file-only backend like Agg makes figures perfectly well and simply
    # shows nothing, so creating one successfully proves nothing about
    # whether the caller will ever see a frame. Without this the flag would
    # silently degrade into a slower headless run.
    backend = matplotlib.get_backend()
    backend_name = backend.lower()
    is_webagg = "webagg" in backend_name
    if backend_name not in interactive_backends() or is_webagg:
        webagg = (" WebAgg is also refused because matplotlib implements "
                  "pause() there as a blocking server loop, so the worker's "
                  "later frames and outputs are never consumed."
                  if is_webagg else "")
        raise RuntimeError(
            "--live needs an interactive matplotlib backend; this process "
            "has %r.%s Install the Qt backend with "
            "`pip install -e \".[live]\"`, or select another local GUI "
            "backend; otherwise drop --live and read the rendered --out. "
            "The drive is identical either way." % (backend, webagg))
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=figsize or DECLARED_LIVE_FIGSIZE)

    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def worker():
        try:
            outcome["result"] = driver(
                config, progress=progress, on_frame=publish,
                on_node=on_node, on_setup=on_setup,
                stop_requested=stop.is_set)
        except BaseException as exc:            # re-raised on the main thread
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name=thread_name)
    thread.start()
    # The live path stabilizes through the SAME chained step a headless render
    # walks, advanced once per unit as it completes. Frames are published in
    # index order, so the state sees exactly the sequence `stabilize` would
    # feed it and the two produce identical placements. Accumulated here
    # rather than precomputed because there is no finished run to walk yet.
    state = StableLayout()
    placed = []
    main_error = None
    try:
        while True:
            try:
                index = ready.get_nowait()
            except queue.Empty:
                # POLLED, never blocked. A blocking `get` parks the main thread for
                # the whole of an engine unit, and for that entire interval the GUI
                # event loop is never serviced -- no redraw, no input -- which the
                # desktop reports as a hung application. The worker is computing
                # with the GIL released, so there is nothing to gain by sleeping in
                # the queue rather than in the event loop.
                #
                # `plt.pause` both pumps the event loop and sleeps for the
                # interval, so this waits at LIVE_POLL_INTERVAL without spinning,
                # whether or not the backend has an event loop of its own.
                plt.pause(LIVE_POLL_INTERVAL)
                continue
            if index is None:
                break
            frames = published["frames"]
            while len(placed) <= index:
                placed.append(place_frame(state, frames[len(placed)]))
            drawer(figure, frames, index, placed)
            figure.canvas.draw_idle()
            # Yields to the GUI event loop; a backend without one still returns.
            plt.pause(LIVE_POLL_INTERVAL)
    except BaseException as error:
        # Never inspect or serialize the node while its worker can still mutate
        # it. The event is observed between the two engine stages and units;
        # joining makes interrupted geometry a stable snapshot.
        main_error = error
        stop.set()
    finally:
        thread.join()
        plt.close(figure)
    if main_error is not None:
        raise main_error
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]

def render(frames, path, drawer, figsize=None):
    """Render frames with ``drawer`` to a GIF, MP4, or final-frame PNG.

    `figsize` defaults to this module's own. An example with a smaller panel
    grid passes its own so its panels keep a sane aspect instead of being
    stretched across a canvas proportioned for the emergence instrument.
    """
    if not frames:
        raise ValueError("cannot render an empty frame sequence")
    lowered = os.fspath(path).lower()
    if not lowered.endswith((".gif", ".mp4", ".png")):
        raise ValueError("--out must end in .gif, .mp4, or .png; got %r"
                         % os.fspath(path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=figsize or DECLARED_RENDER_FIGSIZE)
    # Computed once, in frame order: the alignment is a chain, so a renderer
    # that redraws a frame or draws only the last must still see the same
    # positions it would have seen drawing them all in sequence.
    try:
        placed = stabilize(frames)
        if lowered.endswith(".png"):
            drawer(figure, frames, len(frames) - 1, placed)
            figure.savefig(path, dpi=110)
            return path
        import matplotlib.animation as animation

        def update(index):
            drawer(figure, frames, index, placed)
            return []

        movie = animation.FuncAnimation(figure, update, frames=len(frames),
                                        interval=900, blit=False, repeat=False)
        writer = "pillow" if lowered.endswith(".gif") else "ffmpeg"
        movie.save(path, writer=writer, dpi=100)
    finally:
        plt.close(figure)
    return path


def stabilize(frames):
    """Every frame's stabilized positions and view, computed once in order.

    Precomputed rather than accumulated during drawing because the alignment
    is a chain: each frame is aligned to the one before it. A renderer that
    redraws a frame, or draws only the last, would otherwise get a different
    picture depending on what it had drawn before.

    Returns one entry per frame, `None` where the layout itself is absent.
    """
    state = StableLayout()
    return [place_frame(state, frame) for frame in frames]
