"""Figures for the Colab notebook: the shapes, the anchors, and where the error lies.

Distances are measured in the frame the paper scores in: the reference mesh normalised
to [-1, 1], and each prediction aligned onto it by `align_prediction_to_gt`, as `score`
does. For display, everything is then turned into the frame the input image was
rendered in, so that the camera-side view looks like the image. Two viewpoints: that
camera, and the exact opposite direction, which sees what the image does not. Surface
colour is the distance to the reference surface in % of its bounding-box diagonal, the
unit of F@2: blue is under 2 and counts as correct, red is over.

The colours come from a fresh run of the same alignment, which samples points at
random. On some objects that alone moves F@2 by several points: re-scoring the same two
power-drill meshes eight times gave 65.8 +/- 2.1 unanchored and 78.1 +/- 1.5 anchored.

open3d is imported inside the functions that need it, so the statistics figures still
work on a runtime without it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# tools/render_blender.py puts the camera at CAM_LOC in Blender's Z-up world, and
# Blender's OBJ importer brings the Y-up mesh into that world as (x, y, z) -> (x, -z, y).
# Checked against the input images of four objects; pose_from_silhouette.py uses the
# same matrix. Without it the camera-side view is 68 degrees away from the image.
CAM_LOC = np.array([2.2, -2.2, 1.8])
BLENDER_IMPORT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
ERR_NORM = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=2.0, vmax=6.0)
ERR_CMAP = plt.get_cmap("RdBu_r")
ERR_LABEL = "distance to the reference, % of diagonal\n(F@2 counts a point as correct under 2)"
GREY = np.array([0.80, 0.80, 0.82])
ACCENT, MUTED = "#c0392b", "#7f8c8d"


# --- 3D ---------------------------------------------------------------------------------

def to_render(x):
    """A mesh or (N, 3) array from the dataset frame into the render frame (a rotation)."""
    if isinstance(x, np.ndarray):
        return x @ BLENDER_IMPORT.T
    import copy
    m = copy.deepcopy(x)
    m.rotate(BLENDER_IMPORT, center=(0.0, 0.0, 0.0))
    return m


def views():
    """(label, elev, azim) in the render frame: the input camera, then its exact opposite."""
    d = CAM_LOC / np.linalg.norm(CAM_LOC)
    elev, azim = np.degrees(np.arcsin(d[2])), np.degrees(np.arctan2(d[1], d[0]))
    return [("input camera", elev, azim), ("opposite side", -elev, azim + 180.0)]


def hidden_side(reference, n=20000):
    """Reference surface points the input camera cannot see, in the render frame.

    Same hidden-point removal as analyze_occlusion, but from the image's real camera:
    CAM_LOC in the render frame, scaled by 2.0 / 1.3 because the renderer normalised the
    object to 1.3 where the dataset frame uses 2.0."""
    pcd = reference.sample_points_uniformly(number_of_points=n)
    pts = np.asarray(pcd.points)
    _, idx = pcd.hidden_point_removal(CAM_LOC * (2.0 / 1.3), np.linalg.norm(pts.max(0) - pts.min(0)) * 100)
    seen = np.zeros(len(pts), bool)
    seen[np.asarray(idx)] = True
    return pts[~seen]


def recall_pct(points, mesh, diag, thr_pct=2.0):
    """Share of `points` within thr_pct % of `diag` of the surface of `mesh`."""
    import open3d as o3d
    d = _scene(mesh).compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()
    return 100.0 * float((d < thr_pct / 100.0 * diag).mean())


def _direction(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _scene(mesh):
    import open3d as o3d
    # YCB meshes carry several materials; the tensor conversion warns once per call.
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def distance_pct(points, reference):
    """Unsigned distance from each point to the reference surface, in % of its diagonal."""
    import open3d as o3d
    d = _scene(reference).compute_distance(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()
    return 100.0 * d / np.linalg.norm(reference.get_axis_aligned_bounding_box().get_extent())


def decimate(mesh, n_tri=12000):
    return mesh.simplify_quadric_decimation(n_tri) if len(mesh.triangles) > n_tri else mesh


def facing(points, reference, elev, azim, eps=1e-3):
    """Points of the reference surface that a viewer at (elev, azim) can see.
    `points` and `reference` must both be in the render frame."""
    import open3d as o3d
    d = _direction(elev, azim).astype(np.float32)
    P = np.asarray(points, dtype=np.float32)
    rays = np.hstack([P + eps * d, np.tile(d, (len(P), 1))])
    hit = _scene(reference).cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return ~np.isfinite(hit)


def _axes(ax, elev, azim, lim):
    ax.set_proj_type("ortho")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()
    ax.computed_zorder = False               # points added after the mesh stay on top


def draw_mesh(ax, mesh, elev, azim, err=None):
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.45 + 0.55 * np.abs(n @ _direction(elev, azim))   # headlight
    rgb = (np.tile(GREY, (len(F), 1)) if err is None
           else ERR_CMAP(ERR_NORM(err[F].mean(1)))[:, :3])
    ax.add_collection3d(Poly3DCollection(T, facecolors=np.clip(rgb * shade[:, None], 0, 1),
                                         edgecolors="none", linewidths=0))


def draw_points(ax, P, seen, normals=None, size=5):
    if not seen.any():
        return
    ax.scatter(*P[seen].T, s=size, c="k", depthshade=False)
    if normals is not None:
        ax.quiver(*P[seen].T, *normals[seen].T, length=0.35, color="k", linewidth=0.9)


def _colorbar(fig, cax):
    sm = plt.cm.ScalarMappable(norm=ERR_NORM, cmap=ERR_CMAP)
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1, 2, 4, 6])
    cb.set_label(ERR_LABEL, fontsize=8)
    cb.ax.tick_params(labelsize=8)


def generation_figure(image, reference, aligned, contacts, scores, trace=None, name=""):
    """Input image, reference with its contacts, and each prediction coloured by error.

    aligned: {"unanchored": mesh, "anchored": mesh}, already aligned onto `reference`.
    scores:  {"unanchored": score(...), "anchored": score(...)}.
    """
    import matplotlib.image as mpimg
    preds = {k: decimate(m) for k, m in aligned.items()}
    err = {k: distance_pct(np.asarray(m.vertices), reference) for k, m in preds.items()}
    reference, contacts = to_render(reference), to_render(np.asarray(contacts, dtype=float))
    ref, preds = decimate(reference), {k: to_render(m) for k, m in preds.items()}
    hidden = hidden_side(reference)
    diag = np.linalg.norm(reference.get_axis_aligned_bounding_box().get_extent())
    rec = {k: recall_pct(hidden, to_render(m), diag) for k, m in aligned.items()}
    lim = 1.05 * max(1.0, *(np.abs(np.asarray(m.vertices)).max() for m in preds.values()))

    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.05, 1, 1, 1, 0.045], wspace=0.02, hspace=0.12)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(mpimg.imread(str(image))); ax.set_axis_off()
    ax.set_title("the only input: one image", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    if trace:
        sdf = [t[1] for t in trace]
        ax.plot(range(1, len(sdf) + 1), sdf, "-o", ms=3, c=ACCENT)
        ax.set_xlabel("denoising step", fontsize=9); ax.set_ylabel("mean |SDF| at the contacts", fontsize=9)
        ax.set_title("the constraint being satisfied", fontsize=10)
        ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.set_axis_off()

    f0 = scores["unanchored"]["f2"]
    heads = [("reference + contacts", None)] + [
        (f"{k}   F@2 {scores[k]['f2']:.1f}" + ("" if k == "unanchored" else f" ({scores[k]['f2'] - f0:+.1f})"), k)
        for k in preds]
    for r, (side, elev, azim) in enumerate(views()):
        seen = facing(contacts, reference, elev, azim)
        for c, (head, k) in enumerate(heads, start=1):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            _axes(ax, elev, azim, lim)
            draw_mesh(ax, ref if k is None else preds[k], elev, azim, None if k is None else err[k])
            draw_points(ax, contacts, seen)
            if r == 0:
                ax.set_title(head, fontsize=10)
            elif k is not None:
                ax.set_title(f"recall of what the camera cannot see: {rec[k]:.0f} %", fontsize=9, y=-0.02)
            if c == 1:
                ax.text2D(-0.02, 0.5, side, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
    _colorbar(fig, fig.add_subplot(gs[:, 4]))
    fig.suptitle(f"{name}: black dots are the contacts facing each viewpoint", fontsize=11, y=0.99)
    plt.show()


def poisson_raw(pos, nrm, depth):
    """Screened Poisson from oriented contacts, left in the contact frame (not rescaled)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos.astype(np.float64)))
    pc.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, linear_fit=False)
    return mesh


def poisson_figure(reference, sets, name=""):
    """sets: {label: (pos, nrm, depth, F@2 as scored)}. Surfaces are drawn where Poisson puts
    them, in the reference frame the contacts live in; the score was taken after alignment."""
    surf = {lab: decimate(poisson_raw(p, n, d)) for lab, (p, n, d, _) in sets.items()}
    err = {lab: distance_pct(np.asarray(m.vertices), reference) for lab, m in surf.items()}
    reference = to_render(reference)
    surf = {lab: to_render(m) for lab, m in surf.items()}
    sets = {lab: (to_render(np.asarray(p, float)), to_render(np.asarray(n, float)), d, f2)
            for lab, (p, n, d, f2) in sets.items()}
    ref = decimate(reference)
    lim = 1.05 * max(1.0, *(np.abs(np.asarray(m.vertices)).max() for m in surf.values()))
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 2 + len(sets), width_ratios=[1] * (1 + len(sets)) + [0.045], wspace=0.02, hspace=0.08)
    first = next(iter(sets.values()))
    for r, (side, elev, azim) in enumerate(views()):
        ax = fig.add_subplot(gs[r, 0], projection="3d"); _axes(ax, elev, azim, lim)
        draw_mesh(ax, ref, elev, azim)
        draw_points(ax, first[0], facing(first[0], reference, elev, azim), first[1], size=18)
        if r == 0:
            ax.set_title(f"reference + {len(first[0])} oriented contacts", fontsize=10)
        ax.text2D(-0.02, 0.5, side, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
        for c, (lab, (p, n, d, f2)) in enumerate(sets.items(), start=1):
            ax = fig.add_subplot(gs[r, c], projection="3d"); _axes(ax, elev, azim, lim)
            draw_mesh(ax, surf[lab], elev, azim, err[lab])
            draw_points(ax, p, facing(p, reference, elev, azim), size=18)
            if r == 0:
                ax.set_title(f"Poisson from {lab} contacts   F@2 {f2:.1f} *", fontsize=10)
    _colorbar(fig, fig.add_subplot(gs[:, -1]))
    fig.suptitle(f"{name}: what the contacts alone determine. Surfaces are drawn where Poisson puts them;\n"
                 "* the score is taken after rescaling them onto the reference, as for every prediction",
                 fontsize=10, y=1.0)
    plt.show()


def interactive(reference, aligned, contacts=None, n_tri=40000):
    """One rotatable scene; click the legend to show or hide each surface."""
    import plotly.graph_objects as go
    xs = np.linspace(0, 1, 13)
    scale = [[x, mcolors.to_hex(ERR_CMAP(ERR_NORM(6 * x)))] for x in xs]
    ref = to_render(decimate(reference, n_tri))
    V, F = np.asarray(ref.vertices), np.asarray(ref.triangles)
    data = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                      color="lightgrey", opacity=0.25, name="reference", showlegend=True)]
    for idx, (k, m) in enumerate(aligned.items()):
        m = decimate(m, n_tri)
        err = distance_pct(np.asarray(m.vertices), reference)
        m = to_render(m)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        data.append(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
            intensity=err, cmin=0, cmax=6, colorscale=scale,
            showscale=idx == 0, colorbar=dict(title="% diag.", len=0.6), name=k, showlegend=True,
            visible=True if idx == len(aligned) - 1 else "legendonly"))
    if contacts is not None:
        contacts = to_render(np.asarray(contacts, dtype=float))
        data.append(go.Scatter3d(x=contacts[:, 0], y=contacts[:, 1], z=contacts[:, 2], mode="markers",
                                 marker=dict(size=2.5, color="black"), name="contacts"))
    eye = CAM_LOC / np.linalg.norm(CAM_LOC) * 1.9
    fig = go.Figure(data)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(itemclick="toggle", x=0.01, y=0.98),
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1))),
                      title=dict(text="drag to rotate; the initial view is the input camera", font=dict(size=12)))
    try:
        import google.colab  # noqa: F401  -- a Colab kernel, possibly driven from VS Code
        fig.show(renderer="colab+vscode")
    except ImportError:
        fig.show()


# --- statistics -------------------------------------------------------------------------

def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def frame_bottleneck(estimated, given):
    """Per-case gain from anchoring, with the frame estimated vs given by an oracle."""
    estimated, given = np.asarray(estimated), np.asarray(given)
    lo, hi = np.floor(min(estimated.min(), given.min())), np.ceil(max(estimated.max(), given.max()))
    bins = np.arange(lo - 1, hi + 3, 2)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for v, col, lab in [(estimated, MUTED, "frame estimated from the image"),
                        (given, ACCENT, "frame given by an oracle (upper bound)")]:
        ax.hist(v, bins, color=col, alpha=0.55, label=f"{lab}: mean {v.mean():+.2f}, n = {len(v)}")
        ax.axvline(v.mean(), color=col, ls="--", lw=1.4)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("gain from 8 contact patches, F@2 points (one bar per 2 points)", fontsize=9)
    ax.set_ylabel("cases", fontsize=9)
    ax.set_title("Same contacts, same objects, same seeds: the gap is the output frame", fontsize=10)
    ax.legend(frameon=False, fontsize=8); _clean(ax); plt.show()


def route_bimodality(gain):
    """Per-case gain of depth reprojection over the image alone."""
    gain = np.asarray(gain)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.arange(np.floor(gain.min() / 4) * 4, np.ceil(gain.max() / 4) * 4 + 4, 4)
    ax.hist(gain[gain > 0], bins, color=ACCENT, alpha=0.7, label=f"improved: {(gain > 0).sum()} cases, mean {gain[gain > 0].mean():+.1f}")
    ax.hist(gain[gain < 0], bins, color=MUTED, alpha=0.8, label=f"degraded: {(gain < 0).sum()} cases, mean {gain[gain < 0].mean():+.1f}")
    ax.axvline(0, color="k", lw=0.6); ax.axvline(gain.mean(), color="k", ls="--", lw=1.2)
    ax.text(gain.mean(), ax.get_ylim()[1] * 0.95, f" paired mean {gain.mean():+.2f}", fontsize=8, va="top")
    ax.set_xlabel("depth reprojected to 4 views, minus image only, F@2 points", fontsize=9)
    ax.set_ylabel("cases", fontsize=9)
    ax.set_title("A mean gain made of two populations: a route that wins big or loses big", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); _clean(ax); plt.show()


def manipulations(comps, sign_test):
    """Every paired difference behind Figure 2a, not just its mean."""
    fig, ax = plt.subplots(figsize=(8, 3.9))
    rng = np.random.default_rng(0)
    for row, (lab, v, _) in enumerate(comps[::-1]):
        v = np.asarray(v); k, n, p = sign_test(v)
        col = ACCENT if p < 0.05 else MUTED
        ax.scatter(v, row + rng.uniform(-0.18, 0.18, len(v)), s=12, color=col, alpha=0.55, lw=0)
        ax.plot([v.mean()] * 2, [row - 0.3, row + 0.3], color=col, lw=2.4)
        ax.text(1.01, row, f"{k}/{n}  p={p:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=col)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(comps))); ax.set_yticklabels([c[0] for c in comps[::-1]], fontsize=8)
    ax.set_xlabel("paired difference, F@2 points (dot: one object; bar: mean)", fontsize=9)
    ax.set_title("Red: the sign test clears 0.05. Where the anchors are matters, how many does not", fontsize=10)
    _clean(ax); plt.show()


def pose_spread(angles_deg):
    """Rotation between the output frames of two seeds, against a uniformly random rotation."""
    a = np.asarray(angles_deg)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(a, np.arange(0, 181, 10), density=True, color=ACCENT, alpha=0.6,
            label=f"measured: {len(a)} seed pairs, median {np.median(a):.0f} deg")
    # angle of a Haar-random rotation: density (1 - cos t) / pi, cdf (t - sin t) / pi
    t = np.linspace(0, np.pi, 2001)
    median = np.degrees(t[np.searchsorted(t - np.sin(t), np.pi / 2)])
    ax.plot(np.degrees(t), (1 - np.cos(t)) / 180, color="k", lw=1.4,
            label=f"angle of a uniformly random rotation (median {median:.0f} deg)")
    ax.set_xlim(0, 180); ax.set_xlabel("rotation between the output frames of two seeds, degrees", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("The output pose is drawn with the noise", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); _clean(ax); plt.show()
