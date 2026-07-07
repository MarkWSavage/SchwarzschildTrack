"""
Assemble Kerr black-hole shadow images (one per spin value a) and publish
them as an interactive Plotly figure with a slider over a, saved as a
self-contained HTML file.
"""
import base64
import io
import time
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from raytrace import trace
from kerr_geodesics import isco_radius

# ---------------------------------------------------------------------------
# camera / image-plane setup
# ---------------------------------------------------------------------------
R_OBS = 50.0
THETA_OBS = np.deg2rad(75.0)      # observer inclination from the spin axis
HALF_FOV = 9.0                    # image plane spans [-HALF_FOV, HALF_FOV] in M
RESOLUTION = 500                  # RESOLUTION x RESOLUTION pixels
DISK_OUTER = 20.0                 # outer edge of the accretion disk, in M
DISK_H_RATIO = 0.15                # disk aspect ratio H/r (constant-opening-angle torus)

# peak of the (unshifted) x**-3 * (1 - x**-0.5) emissivity profile, x = r/r_isco
_X_GRID = np.linspace(1.0001, 50.0, 200_000)
_EMISSIVITY_PEAK = (_X_GRID**-3 * (1 - _X_GRID**-0.5)).max()


def blackbody_rgb(temp_k):
    """Tanner Helland's polynomial fit of the Planckian locus to sRGB."""
    t = np.clip(temp_k, 1000.0, 40000.0) / 100.0

    red = np.where(t <= 66, 255.0,
                    329.698727446 * np.power(np.clip(t - 60, 1, None), -0.1332047592))
    green = np.where(
        t <= 66,
        99.4708025861 * np.log(np.clip(t, 1, None)) - 161.1195681661,
        288.1221695283 * np.power(np.clip(t - 60, 1, None), -0.0755148492))
    blue = np.where(t >= 66, 255.0,
                     np.where(t <= 19, 0.0,
                              138.5177312231 * np.log(np.clip(t - 10, 1, None)) - 305.0447927307))

    return np.clip(np.stack([red, green, blue], axis=-1) / 255.0, 0.0, 1.0)


def disk_color(r_disk, g_disk, a):
    """Color and relativistically-beamed brightness of the accretion disk at
    the given impact radii, given the redshift factor g = E_obs/E_emit from
    disk_redshift(). Inner/blueshifted (approaching) regions run hot and
    bright; outer/redshifted (receding) regions run cool and dim.
    """
    r_isco = isco_radius(a)
    x = np.clip(r_disk / r_isco, 1.0001, None)

    emissivity = np.clip(x**-3 * (1 - x**-0.5), 0.0, None) / _EMISSIVITY_PEAK
    t_emit = emissivity**0.25                      # local "temperature", peak = 1

    g = np.clip(g_disk, 0.0, 3.0)
    t_obs = t_emit * g                              # Doppler/gravitational shift of color
    intensity = emissivity * g**4                    # relativistic beaming of flux

    temp_k = 1500.0 + t_obs * 9000.0
    rgb = blackbody_rgb(temp_k)
    return rgb * np.clip(intensity, 0.0, 3.0)[:, None]**0.5


def legend_gradient_css(n=9):
    """CSS linear-gradient stops approximating disk_color's redshift ->
    blackbody mapping at fixed peak emissivity, for the legend swatch."""
    g = np.linspace(0.4, 2.0, n)
    temp_k = 1500.0 + g * 9000.0
    rgb = blackbody_rgb(temp_k) * np.clip(g**4, 0.0, 3.0)[:, None]**0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    stops = [f'rgb({r*255:.0f},{g_*255:.0f},{b*255:.0f})' for r, g_, b in rgb]
    return 'linear-gradient(to right, ' + ', '.join(stops) + ')'


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
    status, theta_f, phi_f, r_disk, g_disk = trace(
        alpha, beta, R_OBS, THETA_OBS, a, disk_outer=DISK_OUTER,
        disk_h_ratio=DISK_H_RATIO)
    if verbose:
        n_abs = (status == 0).sum()
        n_esc = (status == 1).sum()
        n_unr = (status == 2).sum()
        n_disk = (status == 3).sum()
        print(f'a={a:+.3f}  absorbed={n_abs:6d}  escaped={n_esc:6d}  '
              f'disk={n_disk:6d}  unresolved={n_unr:6d}  ({time.time()-t0:.1f}s)')

    rgb = np.zeros((alpha.size, 3))
    esc = (status == 1)
    rgb[esc] = celestial_color(theta_f[esc], phi_f[esc])

    disk = (status == 3)
    rgb[disk] = disk_color(r_disk[disk], g_disk[disk], a)
    # absorbed and unresolved (status 0 and 2) stay pure black (shadow)

    img = rgb.reshape(resolution, resolution, 3)
    return img


def frame_to_data_uri(frame_uint8):
    """Encode an (H, W, 3) uint8 array as a base64 PNG data URI. Plotly's
    go.Image(z=...) serializes every pixel as JSON numbers, which balloons
    the output HTML; a compressed PNG source is far smaller for images that
    are mostly flat black (the shadow) or smooth gradients (the disk/sky).
    """
    buf = io.BytesIO()
    Image.fromarray(frame_uint8, 'RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


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

    data_uris = [frame_to_data_uri(frame) for frame in frames_data]

    fig = go.Figure(
        data=[go.Image(source=data_uris[0])],
        frames=[
            go.Frame(data=[go.Image(source=data_uris[i])], name=f'{a:.3f}')
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

    plot_fragment = fig.to_html(include_plotlyjs=True, full_html=False)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kerr Black Hole Shadow &amp; Accretion Disk</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; padding: 2rem 1rem 3rem;
    background: #0b0b12; color: #d8d8e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }}
  .wrap {{ max-width: 860px; width: 100%; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin: 0 0 .3rem; color: #f2f2f8; }}
  .subtitle {{ color: #9a9aab; font-size: .95rem; margin: 0 0 1.5rem; line-height: 1.5; }}
  .plot-holder {{ display: flex; justify-content: center; }}
  .legend {{
    margin-top: 1.5rem; padding: 1rem 1.25rem; border-radius: 10px;
    background: #14141e; border: 1px solid #242432;
    font-size: .85rem; line-height: 1.6; color: #c2c2d0;
  }}
  .legend h2 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;
                color: #8a8aa0; margin: 0 0 .6rem; }}
  .legend-row {{ display: flex; align-items: center; gap: .75rem; margin: .5rem 0; }}
  .swatch {{ flex: 0 0 auto; width: 18px; height: 18px; border-radius: 4px; }}
  .gradient-bar {{ flex: 1 1 auto; height: 12px; border-radius: 6px; }}
  .gradient-labels {{ display: flex; justify-content: space-between; font-size: .75rem;
                      color: #8a8aa0; margin-top: .2rem; }}
  footer {{ margin-top: 1.5rem; font-size: .75rem; color: #6a6a80; text-align: center; }}
  footer a {{ color: #9a9ac0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Kerr black hole shadow &amp; accretion disk</h1>
  <p class="subtitle">
    Backward ray-traced null geodesics in Kerr spacetime (observer inclination
    {np.rad2deg(THETA_OBS):.0f}&deg; from the spin axis). Drag the slider to see how the
    shadow and disk change with spin <code>a</code>, from Schwarzschild
    (<code>a=0</code>) to near-extremal (<code>a=0.998</code>).
  </p>
  <div class="plot-holder">
  {plot_fragment}
  </div>
  <div class="legend">
    <h2>Color key</h2>
    <div class="legend-row">
      <div class="swatch" style="background:#000;"></div>
      <div>Black hole shadow &mdash; photons captured by the horizon.</div>
    </div>
    <div class="legend-row">
      <div class="swatch" style="background:conic-gradient(red,yellow,lime,cyan,blue,magenta,red);
                                  border-radius:50%;"></div>
      <div>Lensed sky &mdash; a latitude/longitude grid on the celestial sphere. Longitude
      lines are hue-coded; repeated hue bands near the shadow edge show how many times
      that ray wound around the black hole (higher-order photon rings).</div>
    </div>
    <div class="legend-row">
      <div class="gradient-bar" style="background:{legend_gradient_css()};"></div>
    </div>
    <div class="gradient-labels">
      <span>receding &middot; redshifted &middot; dim</span>
      <span>approaching &middot; blueshifted &middot; bright</span>
    </div>
    <div class="legend-row" style="margin-top:.75rem;">
      <div>Accretion disk &mdash; a finite-thickness torus from the ISCO to
      <code>r=20M</code>, colored by blackbody temperature and shifted by the combined
      gravitational + Doppler redshift, so the side co-rotating toward the observer
      renders hotter and brighter than the receding side.</div>
    </div>
  </div>
  <footer>
    <a href="https://github.com/MarkWSavage/SchwarzschildTrack">github.com/MarkWSavage/SchwarzschildTrack</a>
  </footer>
</div>
</body>
</html>
"""
    with open(out_path, 'w') as f:
        f.write(page)
    print('wrote', out_path)
    return out_path


if __name__ == '__main__':
    spins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.998]
    build_html(spins)
