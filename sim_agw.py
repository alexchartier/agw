import numpy as np
import pdb 
from pyglow import pyglow
import datetime as dt
import matplotlib.pyplot as plt
import scipy.interpolate

"""
AGW/TID simulation based on Kirchengast (1996) and Clark (1970)
Inputs: Period, horizontal wavelength, direction, neutral wind disturbance 
        amplitude at lowest height, climatological inputs
Outputs: 3D estimate of:
            1. neutral velocity, temperature and density perturbations
            2. ion velocity, temperature (i and e) and density perturbations
"""

K_B = 1.3805E-16
N_A = 6.022E23
R = K_B * N_A
g = 981.
i = 1.j

lambda_0 = 324  # thermal conductivity coefficient in dyne-cm2 / sec-K (pg. 52)

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

    mlat = 40.
    mlon = 10.
    period = 30.    # Period of wave (minutes).

    phi_H = 0.1    # propagation direction (N = 0, S = 180)
    A = 5.  # magnitude of wind disturbance at the bottom (m/s)

    # Horizontal wavenumbers in cm^-1 
    k = 0.01 * 1E-5
    kx = k * np.cos(np.deg2rad(phi_H))
    ky = k * np.sin(np.deg2rad(phi_H))

    v_n1, p_n1, rho_n1 = agw_perts(time, time, x, y, z, k, kx, ky, \
        mlat, mlon, period, A) 


def agw_perts(time, starttime, x, y, z, k, kx, ky, mlat, mlon, period, A):
    """
    period: GW period in minutes
    Polarization relations: U, V, W, P, R
    Wave equation: A
    Height axis: z
    v defined in x, y, z coordinates (meridional, zonal, vertical)
    """
    omega = 1 / (period * 60) * 2 * np.pi
    t = (time - starttime).total_seconds()
 
    # Background atmospheric stuff 
    T_0, v_nx0, v_ny0, v_in, H, H_dot, rho_0, rho_i, m, p, I, alts = \
        get_bkgd_atmosphere(time, z, mlat, mlon)
    cp, cv, gamma, gamma_1 = get_specific_heat_ratios(T_0)
    A_p = A * np.exp(i * (omega * t - kx * x - ky * y))  # GW equation

    # Various intermediate quantities
    v, omega_p, PSI, k1, c1, c2, c3 = clark_consts(
        lambda_0, T_0, H, omega, k, kx, ky,
        v_nx0, v_ny0, v_in, rho_0, rho_i, I, gamma_1, m, p)
    Kz = calc_Kz_clark(PSI, c1, c2, c3, H, H_dot, k1, omega_p, gamma_1)

    plt.plot(np.imag(Kz), alts)
    plt.show()
    kzr = np.real(Kz)

    # Polarization factors
    P = calc_P(omega_p, gamma, gamma_1, p, Kz, H, H_dot, k1, c2, PSI)
    V = calc_V(omega_p, v_in, ky, g, H, P)
    W = calc_W(omega_p, gamma, p, c1, g, H, c2, PSI, Kz)
    U = calc_U(omega_p, v_in, I, kx, g, H, P, W)

    # velocity, density and pressure perturbations
    v_n1 = [U, V, W] * A_p
    p_n1 = P * A_p
    rho_n1 = R * A_p

    return v_n1, p_n1, rho_n1
             

def calc_brunt_vaisala_freq(z, gamma, H):
    # Brunt-Vaisala freq: sqrt((gamma - 1) * g ** 2 / C **2)
    # C = sqrt(gamma * g * H)

    C = np.sqrt(gamma * g * H)
    omega_B = np.sqrt((gamma - 1) * g ** 2 / C ** 2)
    omega_B /= (2 * np.pi)
    alts = z/1E5 + 80
    plt.plot(1 / (omega_B * 60), alts)
    plt.xlabel('Period (min)')
    plt.ylabel('Alt. (km)')
    plt.show()
    return omega_B
    
   
def calc_U(omega_p, v_ni, I, kx, g, H, P, W):
    """
    Kirchengast (A5)
    omega_p: related to GW frequency (omega_p = w - kx * v_nxo - ky * v_nyo)
    v_ni: ion-neutral collision frequency
    I: magnetic field inclination
    kx: x-component of wavenumber
    g: gravity
    H: scale height
    P: polarization factor
    W: polarization factor
    U: polarization factor
    """
    U = 1 / (omega_p - i * v_ni * np.sin(I) ** 2) * \
            (kx * g * H * P - i * v_ni * np.cos(I) * np.sin(I) * W)
    return U


def calc_V(omega_p, v_ni, ky, g, H, P):
    """
    Kirchengast (A1)
    omega_p: 
    v_ni: ion-neutral collision frequency
    ky: y-component of wavenumber
    g: gravity
    H: scale height
    P: Polarization factor
    V: another polarization factor
    """
    V = 1 / (omega_p - i * v_ni) * (ky * g * H * P)
    return V


def calc_W(omega_p, gamma, p_0, c1, g, H, c2, PSI, Kz):
    """
    Clark (p25)
    W: polarization factor
    """
    W = omega_p ** 2 * (gamma - 1) / np.sqrt(p_0) * \
        (c1 * g * H * (c2 + PSI * (Kz ** 2 +  1/ (4 * H ** 2))) - omega_p)
    return W


def calc_P(omega_p, gamma, gamma_1, P_0, Kz, H, H_dot, k1, c2, PSI):
    # From Clark, P25
    P = omega_p ** 2 * (gamma - 1) / np.sqrt(P_0) * (\
        (Kz - i / (2 * H) - i * k1) * (c2 + PSI * (Kz ** 2 + 1 / (4 * H ** 2))) +\
        i * (1 + gamma_1 * H_dot) / H)
    return P


def calc_R(P, T):
    R = P - T
    return R


def calc_T(omega_p, gamma, P_0, c1, g, gamma_1, H_dot, Kz, H, K_1):
    T = omega_p * (gamma - 1) / np.sqrt(P_0) * (i * c1 * g * \
        (1 + gamma_1 * H_dot) + omega_p * (Kz - i / (2 * H) - i * K_1))
    return T

def calc_Kz_clark(PSI, c1, c2, c3, H, H_dot, k1, omega_p, gamma_1):
    """
    Kz following Clark (p. 24-25)
    """
    d1 = c2 / PSI + 1 / (2 * H ** 2) + k1 ** 2 - c1 * c3 
    d2 = 1 / (4 * H ** 2) * (1 / (4 * H ** 2) - k1 ** 2 - c1 * c3) + \
        1 / PSI * (c2 / (4 * H ** 2) - c1 * c2 * c3 - c2 * k1 ** 2 - 1 / H ** 2\
        + omega_p * c3 / (g * H) + c1 * g / (omega_p * H) + \
        gamma_1 * H_dot / H * (-1 / (2 * H) + k1 + c1 * g / omega_p))

    Kz = (- d1 / 2 - (d1 ** 2 / 2 - 4 * d2) ** (1 / 2)) ** (1 / 2)

    return Kz


def calc_Kz_volland(gamma, omega, omega_B, kx, M, P_0, T, lambda_0, cp, alts, ):
    """
     From Volland (1969)
     C: Acoustic phase velocity
     gamma: ratio of specific heats
     R: Ideal gas constant
     M: molecular weight
     T: Neutral temperature
     V: heat conduction velocity
     lambda_0: coefficient of heat conductivity
     cp: Specific heat capacity
     P_0: pressure
    
    """
    C = (gamma * (R / M) * T) ** (1 / 2)
    V = lambda_0 * g / (cp * P_0)
    omega_a = gamma * g / (2 * C)
    omega_h = gamma * g / (2 * V)
    k = omega / C
    S = kx / k 
    A = omega_a / omega
    G = omega_h / omega 
    B = 2 * (gamma - 1) ** (1 / 2) / gamma 
    Kz = (gamma / 2 - A ** 2 - S ** 2 - i * G \
        - ((gamma / 2 - i * G) ** 2 + 2 * i * G * (1 + B ** 2 * S ** 2)) ** (1 / 2)
        ) ** (1 / 2)

    plot_Kz_quantities(alts, C, V, omega, omega_a, omega_h, omega_B)

    return Kz

def plot_Kz_quantities(alts, C, V, omega, omega_a, omega_h, omega_B):
    nplts = 3
    fig = plt.figure()
    # C
    ax = fig.add_subplot(1, nplts, 1)
    ax.set_xscale('log')
    ax.plot(C, alts, label='C term')
    ax.legend()
    ax.grid()
    ax.set_ylabel(r'Alt. $(km)$')

    # V
    ax = fig.add_subplot(1, nplts, 2)
    ax.set_xscale('log')
    ax.plot(V, alts, label='V term')
    ax.legend()
    ax.grid()
    ax.set_ylabel(r'Alt. $(km)$')

    # omega
    ax = fig.add_subplot(1, nplts, 3)
    ax.set_xscale('log')
    ax.plot(omega_a, alts, label='omega_a')
    ax.plot(omega_h, alts, label='omega_H')
    ax.plot(omega_B, alts, label='omega_B')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'frequency (Hz)')

    plt.show()


def facalg(AMPL, ):
    """
    See page 110 of Clark thesis - calculates Kz
    # Translation from the outside world
    PSI: THERMK
    lambda_0: THERMC

    """
    # "DATA"
    H = i
    COI = 0.84992E-9
    RAD = 57.296
    GAMMA = 1.4
    

    # pythonized version of Fortran
    if AMPL < 0:
        AMPL = AMPL0
        G1 = GAMMA - 1.
        G2 = GAMMA / G1
        #IS = 0
        WVN = WVN * HCON  
        HNSCM = HNSTEP / HCON
        DELTIM = PERIOD / 1200.
        SINISQ = SINI ** 2
        CSINI = COSI * SINI
        WVNSQ = WVN ** 2
        PHI = PHI / RAD
        WVNX = WVN * np.cos(PHI)
        WVNXSQ = WVNX ** 2
        WVNY = WVN * sin(PHI)
        WVNYSQ = WVNSQ - WVNXSQ
        FREQ = .104718 / PERIOD  # 2 pi factor 
    AMLPHF = 0.
    #IS += 1 
    #JJ = J + 1
    #DO 3 K = 1, JJ 
    G = GC[K]
    HKK = HK[K] / HCON
    GHK = G * HKK
    DENK = DEN[K]
    TK = TEMP[K]
    HK12 = 0.5 / HKK
    FKSQ = 1 ./ (HKK * HKK)
    HKSQ4 = HKSQ / 4.
    COLLIF = COI * DENK
    PRESS = DENN[K] * GHK
    G2H = G2 * GRADH[K]
    GIH = 1. + G2H
    FRQ = FREQ - VNXO[K] * WVNX - VNYO[K] * WVNY
    FRTIM = FRQ * TIME * 3600.
    TDEL0 = WVNX / (FRQ * 3600.)
    PSIO = -Y * WVNY + FRTIM
    FRQSQ = FRQ ** 2
    WGH = FRQ / GHK
    F1 = H * COLLIF
    F2 = FRQ - F1
    F3 = FRQ - F1 * SINISQ
    PRESQ = 1. / np.sqrt(PRESS)
    THERMK = -H * THERMC * (TK ** 2.5) ./ ((TK + 245.4) * PRESS * FRQ)
    THERM2 = 2. * THERMK
    Z1 = WVNX * COLLIF * CSINI / F3
    C1 = WGH - WVNXSQ / F3 - WVNYSQ / F2
    C2 = G2 + WVNSQ * THERMK
    C3 = FRQ * F2 / F3
    W2GH = FRQ * WGH
    GKWH = WVNSQ * HKSQ / W2GH
    Z4 = HK12 / HKK - W2GH + WVNSQ
    Z6 = C2 + THERMK * HKSQ4

    if TIME == TINT:
        if K <= 1:
            WVNZ = - np.sqrt(np.complex(-WVNSQ - HKSQ4 + W2GH / GAMMA + GKWH / G2, 0.))
            WVNT = WVNZ.copy()
        BB = C2 + THERMK * Z4
        CC = Z6 * (HKSQ4 + WVNSQ - W2GH) + G2H * (HKI2 / HKK - GKWH + H * WVNT / HKK) - GKWH + W2GH
        WVNT = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4.* CC * THERMK)) / THERM2)
    WVNZTI[K] = np.imag(WVNT)
    WVNZTR[K] = np.real(WVNT)
    else:
        WVNZ = WVNZR[K] + H * WVNZI[K]
    Z5 = HKSQ4 - Z1 ** 2 - C1 * C3 - 2. * H * Z1 * WVNT
    BB = C2 + THERMK * (Z5 + HKSQ4)
    CC = Z6 * Z5 + G2H * (Z1 - HKI2 + H * WVNZ) / HKK + G1H * C1 * HKSQ / WGH - HKSQ + WGH * C3
    WVNZ = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4. * CC * THERMK)) / THERM2)
    WVNZR[K] = np.real(WVNZ)
    WVNZI[K] = np.imag(WVNZ)
    PSIO = PSIO - WVNZ * HNSCM
    PSIK = PSIO - WVNX * X[K]
    Z2 = (WVNZ * WVNZ + HKSQ4) * THERMK
    Z3 = WVNZ - H * (HKI2 + Z1)
    C4 = FRQSQ * G1 * PRESQ
    PPK = C4 * (Z3 * (G2 + Z2) + H * G1H / HKK)
    PZK = C4 * (C1 * GHK * (C2 + Z2) - FRQ)
    PTK = C4 * (H * C1 * G * G1H / FRQ + Z3)
    PNK = PPK - PTK
    PXK = (WVNX * GHK * PPK + COLLIF * CSINI * PZK) / F3
    TDELAY[K] = TDELO * X[K]
    FACT5 = AMPL * np.exp( H * PSIK)
    AMPGW[K] = .1 + np.real(FACT5 * PNK)
    PZ[K] = PZ[K] + COSI * np.real(FACT5 * PXK) + SINI * np.real(FACT5 * PZK)
    PX[K] = PX[K] / AMPGW[K]
    PT[K] = PT[K] * (1. + np.real(FACT5 * PTK))
    if K != 1:
        PN[K - 1] = (PT[K] - PT[K - 1]) / HNSCM
    if IS == 10:
        EFLUX[K] = .5 * PRESS * np.real(PZK * FACT5 * np.conj(PPK * FACT5))
        AMPLG[K] = np.abs(FACT5) * PRESQ
        AMPLHF = AMPLHF + HNSCM * WVNZTI[K]
        AMPLH[K] = AMPL * np.exp(AMPLHF) * PRESQ

    PN[JJ] = PN[J]    
    










 

    FRQ = omega - v_nx0 * kx - v_ny0 * ky
    THERMK = -i * THERMC * (T_0 ** 2.5) / ((T_0 + 245.4) * p * FRQ)  

    WGH = omega_p / (g * H)
    C1 = WGH - WVNXSQ / F3 - WVNYSQ / F2
    C2 = G2 + WVNSQ * PSI
    C3 = FRQ * F2 / F3


    PSI = THERMK

    return v, omega_p, PSI, k1, c1, c2, c3


def clark_consts(lambda_0, T_0, H, omega, k, kx, ky, v_nx0, v_ny0, v_in, \
        rho_0, rho_i, I, gamma_1, m, p):
    """ 
         Clark, P23
    lambda_0: Unperturbed thermal conductivity coefficient
    T_0: unperturbed neutral gas temperature
    omega: GW frequency 
    kx, ky: GW wavenumbers in the x- and y-dirs (NOTE: y points mag. WEST)
    v_nxo, v_nyo: neutral winds in the x and y dirs at the base
    v_in: ion-neutral coll. freq.
    rho_0, rho_i: ion and neutral densities
    I: dip angle (+ve up)
    gamma_1: related to ratio of specific heats
    PSI: 
    """ 
    P_0 = rho_0 * K_B * T_0 / m
    v = v_in * rho_i / rho_0
    omega_p = omega - kx * v_nx0 - ky * v_ny0

    FRQ = omega - v_nx0 * kx - v_ny0 * ky
    PSI = -i * lambda_0 * (T_0 ** 2.5) / ((T_0 + 245.4) * p * FRQ)  
    # PSI = lambda_0 * T_0 / (i * omega * rho_0 * g * H)
    k1 = kx * v * np.cos(I) * np.sin(I) / (omega_p - i * v * np.sin(I) ** 2)
    c1 = omega_p / (g * H) - kx ** 2 / (omega_p - i * v * np.sin(I) ** 2) - \
         ky ** 2 / (omega_p - i * v)
    c2 = gamma_1 * k ** 2 * PSI
    c3 = omega_p * (omega_p - i * v) / (omega_p - i * v * np.sin(I) ** 2)
    return v, omega_p, PSI, k1, c1, c2, c3


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
        idx = (np.abs(spec_heat_table[:, 0] - T)).argmin()
        cp.append(spec_heat_table[idx, 1])
        cv.append(spec_heat_table[idx, 2])
        gamma.append(spec_heat_table[idx, 3])
    gamma = np.array(gamma)
    gamma_1 = gamma / (gamma - 1)
    return np.array(cp), np.array(cv), gamma, gamma_1


def get_bkgd_atmosphere(time, z, lat, lon):
    """
    T_0: neutral temperature
    v_in: ion-neutral collision frequency
    H: scale height
    H_dot: derivative of scale height
    rho: neutral density
    m: average molar mass
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
    ions = {
        'H+': 1,
        'O+': 16,
        'NO+': 30,
        'O2+': 32,
        'HE+': 4,
    }
        
    alts = z / 1E5 + 120

    # Get basic quantities out of MSIS
    T_0, rho_0, rho_i, m, p, N, N_i, v_nx, v_ny = \
        [], [], [], [], [], [], [], [], []  # N: number density

    for alt in alts:
        pt = pyglow.Point(time, lat, lon, alt)
        pt.run_msis()
        pt.run_iri()
        pt.run_hwm14()
        v_nx.append(pt.u * 1E2)
        v_ny.append(pt.v * 1E2)
        T_0.append(pt.Tn_msis)
        rho_0.append(pt.rho) 
        rho_i.append(sum([pt.ni[k] / 1E6 * v for k, v in ions.items()]) / N_A)
        nnd = sum(pt.nn.values())
        nid = sum(pt.ni.values())
        # Molar mass
        m.append(sum([pt.nn[k] * v for k, v in neutrals.items()]) / nnd)
        N.append(nnd)  # neutral number density
        N_i.append(nid)  # ion number density

    # Convert to np arrays
    T_0 = np.array(T_0)
    v_nx = np.array(v_nx)
    v_ny = np.array(v_ny)
    rho_0 = np.array(rho_0)
    rho_i = np.array(rho_i)
    m = np.array(m)
    N = np.array(N)
    N_i = np.array(N_i) / 1E6
    
    # Calculate scale height
    H = K_B * T_0 * N_A / (m * g)
    H_dot = np.gradient(H, z)

    # Magnetic inclination
    pt.run_igrf()
    I = np.deg2rad(pt.dip)

    # pressure
    p = N * K_B * T_0
    v_in = get_v_in(N, N_i, m) 

    # plot_neutral_atmos(alts, N, N_i, rho_0, rho_i, H, H_dot, p, v_in, v_nx, v_ny, T_0)

    return T_0, v_nx, v_ny, v_in, H, H_dot, rho_0, rho_i, m, p, I, alts


def get_v_in(N, N_i, m):
    # Calculate the ion-neutral coll. freq
    return 2.6E-9 * (N + N_i) * m ** (-1 / 2)


def plot_neutral_atmos(alts, N, N_i, rho_0, rho_i, H, H_dot, \
        p, v_in, v_nx, v_ny, T_0):
    nplts = 7
    fig = plt.figure()
    # neutral and charged number density
    ax = fig.add_subplot(1, nplts, 1)
    ax.set_xscale('log')
    ax.plot(N, alts, label='neutral')
    ax.plot(N_i, alts, label='ion')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'number density $(cm^{-3})$')
    ax.set_ylabel(r'Alt. $(km)$')

    # neutral and charged mass density
    ax = fig.add_subplot(1, nplts, 2)
    ax.set_xscale('log')
    ax.plot(rho_0, alts, label='neutral')
    ax.plot(rho_i, alts, label='ion')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'Mass density $(g/cm^3)$')

    # neutral scale height
    ax = fig.add_subplot(1, nplts, 3)
    ax.plot(H / 1E5, alts, label='neutral scale height')
    ax.plot(H_dot / 1E4, alts, label='H_dot x 10')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'Scale height $(km)$')

    # Atmospheric pressure
    ax = fig.add_subplot(1, nplts, 4)
    ax.set_xscale('log')
    ax.plot(p / 1E4, alts, label='neutral pressure')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'pressure $(kPa)$')

    # coll_freq
    ax = fig.add_subplot(1, nplts, 5)
    ax.set_xscale('log')
    ax.plot(v_in, alts, label='ion-neutral coll. freq.')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'collisions per second')

    # Wind speed
    ax = fig.add_subplot(1, nplts, 6)
    ax.plot(v_nx, alts, label='meridional')
    ax.plot(v_ny, alts, label='zonal')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'wind speed (cm/s)')

    # Temp
    ax = fig.add_subplot(1, nplts, 7)
    ax.plot(T_0, alts, label='Neutral temp')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'Temp. (K)')

    plt.show()


if __name__ == '__main__':
    main()
