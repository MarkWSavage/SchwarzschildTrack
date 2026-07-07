"""
Assemble Kerr black-hole shadow images (one per spin value a) and publish
them as an interactive Plotly figure with a slider over a, saved as a
self-contained HTML file.
"""
import time
import numpy as np
import plotly.graph_objects as go

from raytrace import trace

# ---------------------------------------------------------------------------
# camera / image-plane setup
# ---------------------------------------------------------------------------
R_OBS = 50.0
THETA_OBS = np.deg2rad(75.0)      # observer inclination from the spin axis
HALF_FOV = 9.0                    # image plane spans [-HALF_FOV, HALF_FOV] in M
RESOLUTION = 260                  # RESOLUTION x RESOLUTION pixels


def celestial_color(theta_f, phi_f):
    """Latitude/longitude grid on the celestial sphere: white latitude
    lines, hue-coded (by longitude) longitude lines, dark background.
    Hue-coding phi lets you visually count how many times a lensed ray
    winds around the black hole (repeated hue bands near the shadow edge).
    """
    phi_f = np.mod(phi_f, 2 * np.pi)
    n = theta_f.size
    rgb = np.zeros((n, 3))

    # dark navy background
    rgb[:] = [0.03, 0.03, 0.08]

    lat_line = (np.mod(theta_f, np.deg2rad(15)) < np.deg2rad(1.2))
    lon_line = (np.mod(phi_f, np.deg2rad(15)) < np.deg2rad(1.2))

    # hue-coded longitude lines
    hue = phi_f / (2 * np.pi)
    lon_rgb = hsv_to_rgb(hue, np.ones_like(hue), np.ones_like(hue))
    rgb[lon_line] = lon_rgb[lon_line]

    # white latitude lines (drawn after, so they win at intersections)
    rgb[lat_line] = [1.0, 1.0, 1.0]

    return rgb


def hsv_to_rgb(h, s, v):
    h = np.asarray(h)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.zeros((h.size, 3))
    choices = [
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ]
    for k in range(6):
        mask = (i == k)
        rgb[mask] = choices[k][mask]
    return rgb


def make_frame(a, resolution=RESOLUTION, verbose=True):
    lin = np.linspace(-HALF_FOV, HALF_FOV, resolution)
    alpha_grid, beta_grid = np.meshgrid(lin, lin)
    alpha = alpha_grid.ravel()
    beta = beta_grid.ravel()

    t0 = time.time()
    status, theta_f, phi_f = trace(alpha, beta, R_OBS, THETA_OBS, a)
    if verbose:
        n_abs = (status == 0).sum()
        n_esc = (status == 1).sum()
        n_unr = (status == 2).sum()
        print(f'a={a:+.3f}  absorbed={n_abs:6d}  escaped={n_esc:6d}  '
              f'unresolved={n_unr:6d}  ({time.time()-t0:.1f}s)')

    rgb = np.zeros((alpha.size, 3))
    esc = (status == 1)
    rgb[esc] = celestial_color(theta_f[esc], phi_f[esc])
    # absorbed and unresolved (status 0 and 2) stay pure black (shadow)

    img = rgb.reshape(resolution, resolution, 3)
    return img


def build_html(spins, out_path='kerr_shadow.html', resolution=RESOLUTION,
               cache_path='frames_cache.npz'):
    cache = None
    if cache_path and __import__('os').path.exists(cache_path):
        loaded = np.load(cache_path)
        if loaded['resolution'] == resolution and np.array_equal(loaded['spins'], spins):
            cache = loaded['frames']

    if cache is not None:
        frames_data = [cache[i] for i in range(len(spins))]
    else:
        frames_data = []
        for a in spins:
            img = make_frame(a, resolution=resolution)
            frames_data.append((img * 255).astype(np.uint8))
        if cache_path:
            np.savez_compressed(cache_path, frames=np.stack(frames_data),
                                 spins=np.array(spins), resolution=resolution)

    fig = go.Figure(
        data=[go.Image(z=frames_data[0])],
        frames=[
            go.Frame(data=[go.Image(z=frames_data[i])], name=f'{a:.3f}')
            for i, a in enumerate(spins)
        ],
    )

    steps = []
    for i, a in enumerate(spins):
        steps.append(dict(
            method='animate',
            args=[[f'{a:.3f}'],
                  dict(mode='immediate', frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))],
            label=f'a = {a:.3f}',
        ))

    fig.update_layout(
        title=f'Kerr black hole shadow &amp; lensed sky '
              f'(observer inclination {np.rad2deg(THETA_OBS):.0f} deg)',
        sliders=[dict(active=0, currentvalue=dict(prefix='spin '), pad=dict(t=40),
                      steps=steps)],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10),
        template='plotly_dark',
        width=800, height=860,
    )
    fig.update_xaxes(scaleanchor='y', scaleratio=1)

    fig.write_html(out_path, include_plotlyjs=True, full_html=False)
    print('wrote', out_path)
    return out_path


if __name__ == '__main__':
    spins = [0.0, 0.5, 0.9, 0.998]
    build_html(spins)
