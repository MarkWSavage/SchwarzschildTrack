# SchwarzschildTrack

A from-scratch Kerr black hole ray tracer, built in pure Python/NumPy, with an
interactive Plotly visualization of the black hole shadow, lensed sky, and
accretion disk across a range of spin values and observer inclinations.

## How it works

- **`kerr_geodesics.py`** — Derives the null-geodesic equations of motion in
  Kerr spacetime (Boyer-Lindquist coordinates, `G = c = M = 1`) symbolically
  with `sympy`, starting from the geodesic Hamiltonian
  `H = 1/2 g^{mu nu} p_mu p_nu` rather than the textbook Carter radial/angular
  potentials. This form has no square roots in the ODE system, so it
  integrates smoothly through radial/angular turning points with no manual
  sign flips. Also provides the ISCO radius and the combined
  gravitational + Doppler redshift factor for matter on a circular
  equatorial orbit (Bardeen, Press & Teukolsky 1972).
- **`raytrace.py`** — Vectorized backward ray tracer: for every pixel on the
  observer's image plane, a photon is shot backwards from the camera and
  integrated (RK4, adaptive step size) until it either falls through the
  horizon, escapes to the celestial sphere, or hits the accretion disk. All
  pixels are integrated simultaneously as NumPy arrays.
- **`render.py`** — Renders a 2D grid of frames (12 spin values from 0 to
  0.998, x 5 observer inclinations from 15&deg; to 90&deg;, at 500x500
  resolution), with the frame grid computed in parallel across processes
  (`ProcessPoolExecutor`) since each frame is an independent ray trace. The
  celestial sphere is drawn as a latitude/longitude grid (hue-coded by
  longitude, so repeated hue bands near the shadow edge show how many times
  a ray has wound around the hole). The accretion disk is a finite-thickness,
  constant-opening-angle torus spanning the ISCO out to `r = 20M`, colored by
  a blackbody temperature profile shifted by relativistic beaming, so the
  side rotating toward the observer renders hotter and brighter than the
  receding side. Frames are embedded as compressed PNGs (via Pillow) rather
  than raw pixel arrays to keep the output HTML small. Since Plotly's
  built-in slider/frame mechanism only natively drives one animation axis,
  the two sliders (spin, inclination) are plain HTML range inputs wired to
  `Plotly.restyle` via a small embedded JS lookup table.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy sympy plotly pillow

python render.py          # writes kerr_shadow.html (frames cached in frames_cache.npz)
```

Open `kerr_shadow.html` in a browser and drag the sliders to compare shadows
and disk images across spin values and observer inclinations.
