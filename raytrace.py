"""
Vectorized backward ray tracer for Kerr black-hole images.

For every pixel (alpha, beta) on the observer's image plane we shoot a
photon backwards in time from the camera and integrate the Hamiltonian
null-geodesic equations of motion (kerr_geodesics.py) until the ray either

  * falls through the horizon  -> pixel belongs to the black-hole shadow
  * escapes to large radius    -> pixel shows the lensed celestial sphere,
                                   sampled at the ray's final (theta, phi)

All pixels are integrated simultaneously as numpy arrays (a "wavefront" of
~1e5 rays advanced together with an adaptive step size per ray), which is
what makes this fast enough in pure Python/numpy.
"""
import numpy as np
from kerr_geodesics import derivatives, horizon_radius, isco_radius, disk_redshift, \
    f_g_rr_inv, f_g_thth_inv, f_g_tt_inv, f_g_tphi_inv, f_g_phiphi_inv


def initial_conditions(alpha, beta, r_obs, theta_obs, a):
    """Map image-plane impact parameters (alpha, beta) to initial (r, theta,
    phi, p_r, p_theta) at the camera, for photon energy E = 1.

    Standard construction (Bardeen 1973 / Cunningham & Bardeen 1973):
        L = -alpha * sin(theta_obs)
        Q = beta**2 + cos(theta_obs)**2 * (alpha**2 - a**2)
    and one can show Theta(theta_obs) = Q - cos^2(theta_obs)[L^2/sin^2(theta_obs) - a^2]
    reduces exactly to beta**2 at the camera, so p_theta0 = -beta directly.
    p_r0 is fixed (negative root, ray points inward) from the H=0 null
    constraint using the same inverse-metric components as the EOM, so it
    is guaranteed self-consistent with the integrator.
    """
    n = alpha.size
    r0 = np.full(n, r_obs)
    th0 = np.full(n, theta_obs)
    phi0 = np.zeros(n)

    L = -alpha * np.sin(theta_obs)
    pth0 = -beta

    g_rr = f_g_rr_inv(r0, th0, a)
    g_tt = f_g_tt_inv(r0, th0, a)
    g_tp = f_g_tphi_inv(r0, th0, a)
    g_pp = f_g_phiphi_inv(r0, th0, a)
    g_thth = f_g_thth_inv(r0, th0, a)

    E = 1.0
    rhs = -(g_thth * pth0**2 + g_tt * E**2 - 2 * g_tp * E * L + g_pp * L**2) / g_rr
    pr0 = -np.sqrt(np.clip(rhs, 0.0, None))

    return r0, th0, phi0, pr0, pth0, L


def trace(alpha, beta, r_obs, theta_obs, a, max_steps=6000, escape_radius=None,
          step_coeff=0.04, dlambda_max=4.0, disk_outer=20.0, disk_inner=None):
    """Integrate the wavefront of rays. Returns
    (status, theta_f, phi_f, r_disk, g_disk).

    status: 0 = absorbed by horizon, 1 = escaped to celestial sphere,
            2 = unresolved after max_steps (treated as shadow edge / photon ring),
            3 = absorbed by the equatorial accretion disk.
    r_disk, g_disk: radius of disk impact and the combined gravitational +
    Doppler redshift factor there (only meaningful where status == 3).

    The disk is modeled as geometrically thin and optically thick, occupying
    the equatorial plane between disk_inner (default: the ISCO) and
    disk_outer. Set disk_outer=None to disable the disk entirely.
    """
    if escape_radius is None:
        escape_radius = r_obs + 1.0
    if disk_inner is None:
        disk_inner = isco_radius(a)

    r_h = horizon_radius(a)
    r, th, phi, pr, pth, L = initial_conditions(alpha, beta, r_obs, theta_obs, a)

    n = r.size
    status = np.full(n, 2, dtype=np.int8)   # default: unresolved
    theta_f = np.zeros(n)
    phi_f = np.zeros(n)
    r_disk = np.zeros(n)
    g_disk = np.zeros(n)
    active = np.ones(n, dtype=bool)

    eps_th = 1e-6

    old_err = np.seterr(all='ignore')
    for _ in range(max_steps):
        if not active.any():
            break

        def deriv(r_, th_, phi_, pr_, pth_):
            dr, dth, dphi, dpr, dpth = derivatives(r_, th_, pr_, pth_, a, L)
            return dr, dth, dphi, dpr, dpth

        dlam = np.clip(step_coeff * (r - 0.9 * r_h), 1e-4, dlambda_max)

        k1 = deriv(r, th, phi, pr, pth)
        r1 = r + 0.5 * dlam * k1[0]
        th1 = np.clip(th + 0.5 * dlam * k1[1], eps_th, np.pi - eps_th)
        phi1 = phi + 0.5 * dlam * k1[2]
        pr1 = pr + 0.5 * dlam * k1[3]
        pth1 = pth + 0.5 * dlam * k1[4]

        k2 = deriv(r1, th1, phi1, pr1, pth1)
        r2 = r + 0.5 * dlam * k2[0]
        th2 = np.clip(th + 0.5 * dlam * k2[1], eps_th, np.pi - eps_th)
        phi2 = phi + 0.5 * dlam * k2[2]
        pr2 = pr + 0.5 * dlam * k2[3]
        pth2 = pth + 0.5 * dlam * k2[4]

        k3 = deriv(r2, th2, phi2, pr2, pth2)
        r3 = r + dlam * k3[0]
        th3 = np.clip(th + dlam * k3[1], eps_th, np.pi - eps_th)
        phi3 = phi + dlam * k3[2]
        pr3 = pr + dlam * k3[3]
        pth3 = pth + dlam * k3[4]

        k4 = deriv(r3, th3, phi3, pr3, pth3)

        r_new = r + dlam / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        th_new = np.clip(th + dlam / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                          eps_th, np.pi - eps_th)
        phi_new = phi + dlam / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        pr_new = pr + dlam / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        pth_new = pth + dlam / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

        r_old, th_old = r, th
        r = np.where(active, r_new, r)
        th = np.where(active, th_new, th)
        phi = np.where(active, phi_new, phi)
        pr = np.where(active, pr_new, pr)
        pth = np.where(active, pth_new, pth)

        if disk_outer is not None:
            crossed = active & (((th_old - np.pi / 2) * (th - np.pi / 2)) < 0)
            denom = np.where(crossed, th - th_old, 1.0)
            frac = np.where(crossed, (np.pi / 2 - th_old) / denom, 0.0)
            r_cross = r_old + frac * (r - r_old)
            in_disk = crossed & (r_cross >= disk_inner) & (r_cross <= disk_outer)
        else:
            in_disk = np.zeros(n, dtype=bool)

        newly_absorbed = active & ~in_disk & (r <= r_h * 1.001)
        newly_escaped = active & ~in_disk & (r >= escape_radius)

        status[in_disk] = 3
        r_disk[in_disk] = r_cross[in_disk]
        g_disk[in_disk] = disk_redshift(r_cross[in_disk], a, L[in_disk])
        status[newly_absorbed] = 0
        status[newly_escaped] = 1
        theta_f[newly_escaped] = th[newly_escaped]
        phi_f[newly_escaped] = phi[newly_escaped]

        active &= ~(in_disk | newly_absorbed | newly_escaped)

    return status, theta_f, phi_f, r_disk, g_disk
