"""
Symbolic derivation of the null-geodesic equations of motion in Kerr
spacetime (Boyer-Lindquist coordinates, geometrized units G = c = M = 1).

Rather than hand-copy the textbook Carter (1968) radial/angular potentials
R(r), Theta(theta) -- which requires getting several signs right -- we build
the geodesic Hamiltonian

    H(r, theta, p_r, p_theta; a, E, L) = 1/2 g^{mu nu} p_mu p_nu

directly from the (symbolically inverted) Kerr metric and let sympy take
the partial derivatives that give Hamilton's equations:

    dr/dlambda      =  dH/dp_r
    dtheta/dlambda  =  dH/dp_theta
    dphi/dlambda    =  dH/dp_phi        (p_phi = L, conserved... but we
                                          still need dphi/dlambda itself)
    dp_r/dlambda    = -dH/dr
    dp_theta/dlambda = -dH/dtheta

E = -p_t and L = p_phi are conserved along any geodesic because the Kerr
metric is stationary and axisymmetric, so we never need to integrate them.
This formulation has no square roots in the ODE system (unlike the R(r),
Theta(theta) form), so it integrates smoothly through radial/angular
turning points -- no manual sign flips needed.
"""
import numpy as np
import sympy as sp

r, th, a, pr, pth, E, L = sp.symbols('r theta a p_r p_theta E L', real=True)

Sigma = r**2 + a**2 * sp.cos(th)**2
Delta = r**2 - 2*r + a**2
Abig = (r**2 + a**2)**2 - Delta * a**2 * sp.sin(th)**2

g_rr_inv = Delta / Sigma
g_thth_inv = 1 / Sigma
g_tt_inv = -Abig / (Sigma * Delta)
g_tphi_inv = -2 * a * r / (Sigma * Delta)
g_phiphi_inv = (Delta - a**2 * sp.sin(th)**2) / (Sigma * Delta * sp.sin(th)**2)

H = sp.Rational(1, 2) * (
    g_rr_inv * pr**2
    + g_thth_inv * pth**2
    + g_tt_inv * E**2
    - 2 * g_tphi_inv * E * L
    + g_phiphi_inv * L**2
)

_d_r = sp.diff(H, pr)
_d_th = sp.diff(H, pth)
_d_phi = sp.diff(H, L)
_d_pr = -sp.diff(H, r)
_d_pth = -sp.diff(H, th)

_args = (r, th, pr, pth, a, L, E)

f_dr = sp.lambdify(_args, _d_r, 'numpy')
f_dth = sp.lambdify(_args, _d_th, 'numpy')
f_dphi = sp.lambdify(_args, _d_phi, 'numpy')
f_dpr = sp.lambdify(_args, _d_pr, 'numpy')
f_dpth = sp.lambdify(_args, _d_pth, 'numpy')
f_H = sp.lambdify(_args, H, 'numpy')

# also expose the inverse-metric components (needed to set the initial p_r
# from the H = 0 null constraint at the camera)
f_g_rr_inv = sp.lambdify((r, th, a), g_rr_inv, 'numpy')
f_g_thth_inv = sp.lambdify((r, th, a), g_thth_inv, 'numpy')
f_g_tt_inv = sp.lambdify((r, th, a), g_tt_inv, 'numpy')
f_g_tphi_inv = sp.lambdify((r, th, a), g_tphi_inv, 'numpy')
f_g_phiphi_inv = sp.lambdify((r, th, a), g_phiphi_inv, 'numpy')


def derivatives(r_, th_, pr_, pth_, a_, L_, E_=1.0):
    """Right-hand side of Hamilton's equations, vectorized over numpy arrays."""
    args = (r_, th_, pr_, pth_, a_, L_, E_)
    return (f_dr(*args), f_dth(*args), f_dphi(*args), f_dpr(*args), f_dpth(*args))


def horizon_radius(a_):
    return 1.0 + np.sqrt(np.clip(1.0 - a_**2, 0.0, None))


def isco_radius(a_):
    """Innermost stable circular (prograde, equatorial) orbit radius,
    Bardeen-Press-Teukolsky (1972) eq. 2.21. Assumes a_ >= 0."""
    z1 = 1 + (1 - a_**2)**(1/3) * ((1 + a_)**(1/3) + (1 - a_)**(1/3))
    z2 = np.sqrt(3 * a_**2 + z1**2)
    return 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def disk_redshift(r_, a_, L_):
    """Combined gravitational + Doppler redshift factor g = E_obs / E_emit
    for a photon (impact parameter L_, energy E=1) received from matter on a
    prograde circular equatorial Keplerian orbit at radius r_ in Kerr
    spacetime (Bardeen, Press & Teukolsky 1972).
    """
    omega = 1.0 / (r_**1.5 + a_)
    turning = np.clip(r_**1.5 - 3 * np.sqrt(r_) + 2 * a_, 1e-8, None)
    u_t = (r_**1.5 + a_) / (r_**0.75 * np.sqrt(turning))
    denom = u_t * (1 - omega * L_)
    denom = np.where(np.abs(denom) < 1e-8, np.copysign(1e-8, denom), denom)
    return 1.0 / denom


if __name__ == '__main__':
    # sanity check: H should vanish along a null geodesic's initial data,
    # and the equations of motion should reduce to the Schwarzschild
    # (a=0) null geodesic equations in the equatorial plane.
    a0 = 0.0
    r0, th0 = 20.0, np.pi / 2
    L0 = 4.0
    pth0 = 0.0
    g_rr = f_g_rr_inv(r0, th0, a0)
    g_tt = f_g_tt_inv(r0, th0, a0)
    g_pp = f_g_phiphi_inv(r0, th0, a0)
    pr0_sq = -(g_tt * 1.0**2 + g_pp * L0**2) / g_rr
    pr0 = -np.sqrt(pr0_sq)
    print('H at (r0,th0) with constrained pr0 =', f_H(r0, th0, pr0, pth0, a0, L0, 1.0))
    print('dr/dlambda, dtheta/dlambda, dphi/dlambda, dpr/dlambda, dpth/dlambda =',
          derivatives(r0, th0, pr0, pth0, a0, L0, 1.0))
