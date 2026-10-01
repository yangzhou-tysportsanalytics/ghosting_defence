"""Real defence vs ghost overlay animation (Phase 3 task 6), in the spirit of the Grantland ghosts.

Half court (attack to the left, as in the possession tensors). Offence blue (ball handler with a
black edge), ball orange, real defenders red. Team ghost: each slot's mixture components as grey
translucent circles (radius = sigma, opacity ~ weight). Individual ghost of one highlighted
defender: red dashed circles. The ghost at step t uses the offence up to t only (online).

The output shows player positions: keep it private (reports/**/*.mp4 is git-ignored; D-019).
"""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from ghost.viz.court import draw_court  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

OFF, DEF, BALL, GHOST = "#1f77b4", "#d62728", "#ff7f0e", "0.35"


def animate_ghost(
    feats: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    team: dict[str, np.ndarray],
    ind: dict[str, np.ndarray] | None,
    d: int | None,
    path: str | Path,
    title: str = "",
    marks: dict[int, str] | None = None,
    fps: float = 5.0,
) -> Path:
    """feats (11, T, F) unmasked, target (5, T, 2) ft, valid (T,). team / ind: mixture arrays for
    the defender slots, ``w`` (5, T, M) weights, ``mu`` (5, T, M, 2) ft, ``sigma`` (5, T, M);
    ``ind`` holds the individual ghost of slot ``d`` in its row d. marks: step -> label."""
    steps = np.flatnonzero(valid)
    off = np.stack([(feats[1:6, :, 0] + 0.5) * 47.0, (feats[1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(feats[0, :, 0] + 0.5) * 47.0, (feats[0, :, 1] + 0.5) * 50.0], -1)
    handler = feats[1:6, :, 4] > 0.5  # (5, T)
    marks = marks or {}

    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    draw_court(ax, color="0.55")
    ax.set_xlim(0, 47)
    ax.set_ylim(0, 50)
    ax.set_aspect("equal")
    ax.axis("off")
    txt = ax.set_title(title, fontsize=9)
    dyn: list = []

    def frame(k: int):
        for a in dyn:
            a.remove()
        dyn.clear()
        t = steps[k]
        for s in range(5):  # team ghost
            for m in range(team["w"].shape[-1]):
                w = float(team["w"][s, t, m])
                if w < 0.05:
                    continue
                c = Circle(team["mu"][s, t, m], team["sigma"][s, t, m], color=GHOST,
                           alpha=min(0.45, 0.6 * w), lw=0)  # fmt: skip
                dyn.append(ax.add_patch(c))
        if ind is not None and d is not None:
            for m in range(ind["w"].shape[-1]):
                w = float(ind["w"][d, t, m])
                if w < 0.05:
                    continue
                c = Circle(ind["mu"][d, t, m], ind["sigma"][d, t, m], fill=False, ec=DEF,
                           ls="--", lw=1.2, alpha=min(1.0, 0.3 + w))  # fmt: skip
                dyn.append(ax.add_patch(c))
        for j in range(5):
            dyn.append(ax.scatter(*off[j, t], s=70, c=OFF, ec="k" if handler[j, t] else OFF,
                                  lw=1.5, zorder=4))  # fmt: skip
            hl = d is not None and j == d
            dyn.append(ax.scatter(*target[j, t], s=70 if hl else 50, c=DEF,
                                  ec="k" if hl else DEF, lw=1.5, zorder=5))  # fmt: skip
        dyn.append(ax.scatter(*ball[t], s=25, c=BALL, zorder=6))
        lab = marks.get(int(t), "")
        txt.set_text(f"{title}\nt = {t * 0.2:4.1f} s   {lab}")
        return dyn

    anim = animation.FuncAnimation(fig, frame, frames=len(steps), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=110)
    plt.close(fig)
    return path
