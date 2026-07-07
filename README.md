# SchwarzschildTrack

A from-scratch Kerr black hole ray tracer, built in pure Python/NumPy, with an
interactive Plotly visualization of the black hole shadow, lensed sky, and
accretion disk across a range of spin values.

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
- **`render.py`** — Assembles ray-traced frames into an interactive HTML
  figure with a slider over spin `a`. The celestial sphere is drawn as a
  latitude/longitude grid (hue-coded by longitude, so repeated hue bands near
  the shadow edge show how many times a ray has wound around the hole). The
  accretion disk spans the ISCO out to `r = 20M`, colored by a blackbody
  temperature profile shifted by relativistic beaming, so the side rotating
  toward the observer renders hotter and brighter than the receding side.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy sympy plotly

python render.py          # writes kerr_shadow.html (frames cached in frames_cache.npz)
```

Open `kerr_shadow.html` in a browser and drag the slider to compare shadows
and disk images across spin values.
