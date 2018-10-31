import numpy as np
import pdb 
from pyglow import pyglow
import datetime as dt

"""
AGW/TID simulation based on Kirchengast (1996) and Clark (1970)
Inputs: Period, horizontal wavelength, direction, neutral wind disturbance 
        amplitude at lowest height, climatological inputs
Outputs: 3D estimate of:
            1. neutral velocity, temperature and density perturbations
            2. ion velocity, temperature (i and e) and density perturbations
"""

K_B = 0.13805E-5

# Define dimensions (in km)
x = np.linspace(-10000, 10000, 1)
y = np.linspace(-10000, 10000, 1)
z = np.linspace(50, 300, 1)

# TODO(ATC): Check A_0 is the same as quantity in the input table
# TODO(ATC): Check whether x, y, z are defined reasonably
# TODO(ATC): Check whether kx, ky, kz are calculated or defined

def main():
    time = dt.datetime(2010, 1, 1)

    # Coordinates in cm
    x = np.linspace(0, 500, 50) * 1E5
    y = np.linspace(0, 500, 50) * 1E5
    z = (np.linspace(80, 500, 50) - 120) * 1E5  # z = 0 at 120km

    # Horizontal wavenumbers in cm^-1 
    kx = 1E-8
    ky = 1E-8

    mlat = 10
    mlon = 10
    period = 60    # Period of wave (minutes)

    phi_H = 180    # propagation direction (N = 0, S = 180)
    A = 5  # magnitude of wind disturbance at the bottom (m/s)
    

def agw_perts(time, x, y, z, kx, ky, mlat, mlon, period, A):
    """
    period: GW period in minutes
    Polarization relations: U, V, W, P, R
    Wave equation: A
    Height axis: z
    v defined in x, y, z coordinates (meridional, zonal, vertical)
    """
    omega = 1 / (period * 60)
 
    # Background atmospheric stuff 
    T_0, v_in, H, H_dot, M, rho, m, p = get_bkgd_atmosphere(time, z, mlat, mlon)
    cp, cv, gamma, gamma_1 = get_specific_heat_ratios(T_0)
    A_p = A * np.exp(i * (omega * t - kx * x - ky * y))  # GW equation
    lambda_0 = get_thermal_cond_coeffs(T_0)

    # Various intermediate quantities
    v, w_p, PSI, k_1, c_2 = clark_consts(lambda_0, T_0, omega, kx, ky, \
        v_nx0, v_ny0, v_in, rho_0, rho_i, rho_n, I, gamma_1, m)
    H, H_dot = get_scale_height(time)  # or from Clark's equation
    omega_B = calc_brunt_vaisala_freq(T_0, p)
    Kz = calc_Kz(gamma, omega, omega_B, kx, M, lambda_0, cp, P)
    kzr = np.real(Kz)
    a_p = ?

    # Polarization factors
    P = calc_P(w_p, gamma, gamma_1, P_0, Kz, H, H_dot, k_1, c_2, PSI)
    V = calc_V(w_p, v_in, ky, g, H, P)
    W = calc_W(a_p, kzr, z)
    U = calc_U(w_p, v_in, I, kx, g, H, P, W)

    # velocity, density and pressure perturbations
    v_n1 = [U(z), V(z), W(z)] * A_p
    p_n1 = P(z) * A_p
    rho_n1 = R(z) * A_p

    return v_n1, p_n1, rho_n1
             

def calc_brunt_vaisala_freq(T_0, p, p_ref=1E6, R_over_cp=0.286, g=980):
    # From Wikipedia
    # p, p_ref are pressure in dynes/cm2
    # T_0 in K
    # R/cp given by wikipedia
    # g: gravity in cm/s^2
    
    THETA = T_0 * (p_ref / p_0) ** R_over_cp
    omega_B = np.sqrt(g / THETA * THETA_dot)
    return omega_B
    
    
def get_thermal_cond_coeffs(T_0, A_p, T):
    # NOTE: from dynamics of atmospheric reentry p. 37 (Regan)
    lambda_0 = 2.64638E-3 * T ** (3 / 2) / (T + 245.4 * 10 ** (-12 / T)) / 100
    # 100 factor converts from J / s.m.K to J / s.cm.K

    
def calc_U(w_p, v_ni, I, kx, g, H, P, W):
    U = 1 / (w_p - i * v_ni * np.sin(I) ** 2) * \
            (kx * g * H * P - i * v_ni * np.cos(I) * np.sin(I) * W)
    return U


def calc_V(w_p, v_ni, ky, g, H, P):
    V = 1 / (w_p - i * v_ni) * ky * g * H * P)
    return V


def calc_W(a_p, kzr, z):
    """
    a_p: weakly height-dependent factor
    kzr: real component of Kz
    """
    W = a_p * np.exp(-i * kzr * z)
    return W


def calc_P(w_p, gamma, gamma_1, P_0, Kz, H, H_dot, k_1, c_2, PSI):
    # From Clark, P25
    P = w_p ** 2 * (gamma - 1) / np.sqrt(P_0) * (\
        (Kz - i / (2 * H) - i * k_1) * (c_2 + PSI * (Kz ** 2 + 1 / (4 * H ** 2))) +\
        i * (1 + gamma_1 * H_dot) / H)
    return P


def calc_R(P, T)
    R = P - T
    return R


def calc_T(w_p, gamma, P_0, c_1, g, gamma_1, H_dot, Kz, H, K_1):
    T = w_p * (gamma - 1) / np.sqrt(P_0) * (i * c_1 * g * (1 + gamma_1 * H_dot) + \
        w_p * (Kz - i / (2 * H) - i * K_1))
    return T


def calc_Kz(gamma, omega, omega_B, kx, M, P_0, T, lambda_0, cp, R=8.3144598E7):
    # From Volland (1969)
    C = (gamma * (R / M) * T) ** (1 / 2)
    V = lambda_0 * g / (cp * P_0)
    omega_a = gamma * g / (2 * C)
    omega_h = gamma * g / (2 * V)
    k = omega / C
    S = kx / k 
    A = omega_a / omega
    G = omega_h / omega 
    B = 2 * ((gamma - 1) ** (1 / 2) / gamma
    Kz = (gamma / 2 - A ** 2 - S ** 2 - i * G \
        - ((gamma / 2 - i * G) ** 2 + 2 * i * G * (1 + B ** 2 * S ** 2)) ** (1 / 2)
        ) ** (1 / 2)
    return Kz


def calc_a_p(?):
    # Weakly dependent function of something in height


def clark_consts(lambda_0, T_0, omega, kx, ky, v_nx0, v_ny0, v_in, \
        rho_0, rho_i, rho_n, I, gamma_1, m):
    """ 
         Clark, P23
    lambda_0: Unperturbed thermal conductivity coefficient
    T_0: unperturbed neutral gas temperature
    omega: GW frequency 
    kx, ky: GW wavenumbers in the x- and y-dirs (NOTE: y points mag. WEST)
    v_nxo, v_nyo: neutral winds in the x and y dirs at the base
    v_in: ion-neutral coll. freq.
    rho_i, rho_n: ion and neutral densities
    I: dip angle (+ve up)
    gamma_1: related to ratio of specific heats
    PSI: 
    """ 
    P_0 = rho_0 * K_B * T_0 / m
    v = v_in * rho_i / rho_n
    omega_p = omega - kx * v_nx0 - ky * v_ny0
    PSI = lambda_0 * T_0 / (i * omega_p * P_0)
    k_1 = kx * v * np.cos(I) * np.sin(I) / (omega_p - i * v * np.sin(I) ** 2)
    c_2 = gamma_1 * k ** 2 * PSI
    return v, w_p, PSI, k_1, c_2


def get_scale_height(time):
    # Calculate the neutral scale height and its derivative from MSIS or similar
    return H, H_dot


def get_specific_heat_ratios(T_0=500):
    """
    Look up the ratio of specific heats for a given atmospheric temperature (K)
    """
    spec_heat_table = np.array(
        [[250,  1.003, 0.716, 1.401],
         [300,  1.005, 0.718, 1.400],
         [350,  1.008, 0.721, 1.398],
         [400,  1.013, 0.726, 1.395],
         [450,  1.020, 0.733, 1.391],  
         [500,  1.029, 0.742, 1.387],
         [550,  1.040, 0.753, 1.381],
         [600,  1.051, 0.764, 1.376],
         [650,  1.063, 0.776, 1.370],
         [700,  1.075, 0.788, 1.364],
         [750,  1.087, 0.800, 1.359],
         [800,  1.099, 0.812, 1.354],
         [900,  1.121, 0.834, 1.344],
         [1000, 1.142, 0.855, 1.336],
         [1100, 1.155, 0.868, 1.331],
         [1200, 1.173, 0.886, 1.324],
         [1300, 1.190, 0.903, 1.318],
         [1400, 1.204, 0.917, 1.313],
         [1500, 1.216, 0.929, 1.309]]
    )

    cp, cv, gamma, gamma_1 = [], [], [], []
    for T in T_0:
        idx = (np.abs(gamma_table[:, 0] - T)).argmin()
        cp.append(spec_heat_table[idx, 1])
        cv.append(spec_heat_table[idx, 2])
        gamma.append(spec_heat_table[idx, 3])
        pdb.set_trace() # Check this works - gamma should be 1.387
        gamma_1.append(gamma / (gamma - 1))
    return cp, cv, gamma, gamma_1


def get_bkgd_atmosphere(time, z, mlat, mlon):
    """
    T_0: neutral temperature
    v_in: ion-neutral collision frequency
    H: scale height
    H_dot: derivative of scale height
    M: mass mixing ratio?
    rho: neutral density
    m: mass?
    p: pressure
    """
    neutrals = {
        'H': 1,
        'O': 16,
        'N': 14,
        'AR': 40,
        'N2': 28,
        'O2': 32,
        'HE': 4,
    }                

    alts = z / 1E5 + 120

    # Get basic quantities out of MSIS
    T_0, rho, M, p, N = [], [], [], [], []  # N: number density

    for alt in alts:
        pt = pyglow.Point(time, alt, mlat, mlon)
        pt.run_msis()
        T_0.append(pt.Tn_msis)
        rho.append(pt.rho)
        M.append(sum([pt.nn[k] * v for k, v in neutrals.items()]) / pt.rho)  # Molar mass
        N.append(sum([v for v in neutrals.keys()]))  # number density

    # Calculate scale height
    H = []
    for alt in alts:
        rho_i = rho[alts == alt] / np.e
        try:
            H.append(np.interp(rho, alts, rho_i) - alt)
        except:
            H.append(H[-1])
    H_dot = np.diff(H)

    return T_0, v_in, H, H_dot, M, rho, m, p


def get_v_in(time):
    # Calculate the ion-neutral coll. freq
    return v_in
