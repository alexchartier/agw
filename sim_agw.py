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

"""
    old stuff
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

"""


# "DATA"
H = 1.j
COI = 0.84992E-9
RAD = 57.296
GAMMA = 1.4
BC = 1.3805E-16
QPO = 150  # (1E-22 W.s / m2) F10.7
DEN120 = 0.2461E-10  # g/cm3 at 120 km

HCON = 1E-5
def main():

    # Initialize variables

    # For zprofl
    ALT = np.zeros(300)
    GC = np.zeros(300)
    TEMP = np.zeros(300)
    WM = np.zeros(300)
    VNX0 = np.zeros(300)
    VNY0 = np.zeros(300)
    X = np.zeros(300)
    HK = np.zeros(300)
    TE = np.zeros(300)
    TI = np.zeros(300)
    PT = np.zeros(300)
    DENN = np.zeros(300)
    DENO = np.zeros(300)
    BETAL0 = np.zeros(300)
    PX = np.zeros(300)
    PZ = np.zeros(300)
    PN = np.zeros(300)
    GRADH = np.zeros(300)
    GRADHI = np.zeros(300)
    DEN = np.zeros(300)

    # For cofcal
    A = np.zeros(300)
    B = np.zeros(300)
    C = np.zeros(300)
    QP = np.zeros(300)
    BETA = np.zeros(300)
    COLF = np.zeros(300)
    AMPGW = np.zeros(300)

    # Punchcard inputs
    HNSTEP = 2.0  # height step (km)
    DELTME = 0.5  # time increment (hours)
    QP0 = 100     # F10.7
    BETA0 = 0.008  # (1/s)  magnitude of loss coefficient at 120 km
    ALPHA = 1E-8  # (1/cm3 s) recombination coefficient in cm^-3 sec^-1
    FLUX = 1.   # parameter used in computing GAMMA_infinity (ionization flux at top of atmosphere)
    CONTV = 1.# another parameter used in computing GAMMA_infinity (ionization flux at top of atmosphere)
    HL = 700. # upper boundary (km)
    DIP = -70.  # mag. dip angle (degrees)
    DECL = 0.  # Solar declination (degrees)
    RLAT = 40.  # observer's latitude (degrees +N)
    Y = 0.  # dist. from observer (km? +W)
    TXMAX = 1500  # exospheric max. temp (K)
    TXMIN = 1000  # exospheric min. temp (K)
    T120 = 355 # temp at 120 km (K)
    TEEMAX = 2500  # exospheric electron TMAX (K)
    TEEMIN = 1000  # exospheric electron TMIN (K)
    AMPLO = 2E8 # GW amplitude (dyne^(1/2) sec^2)
    THERMC = 324. # thermal conductivity coefficient * 9/5 to allow for "viscous effects"  (dyne cm^2 / sec. K)
    WVN = 0.02  # Horizontal wavenumber (km^-1)
    PERIOD = 20 # GW period (min)
    PHI = 90  # azimuth (degrees)
    TINT = 12.  # Time of start of GW in hours (must be > 12 or the last TERM read, and even multiple of DELTME)
    TERM = 24.  # Time of GW termination in hours (> TINT and even multiple of DELTME)
    # Program stops when TERM > 98.0 

    print('FLUX and CONTV not properly defined - continuing with them set to 1')

    # Compute general program constants
    ALITRN = 300.
    HCON = 1E-5
    DIP = np.deg2rad(DIP)
    BCO = BC / .2656E-22
    SINI = np.sin(DIP)
    COSI = np.cos(DIP)
    AJ = (HL - 120.) / HNSTEP #  (HL - 120.) / HNSTEP + 1.
    J = int(AJ)
    VOB = 0.
    IPRT= 0
    ITER = 500
    TIME = 12.
    AMPL = 0.
    TERM = 11.50
    DECLR = np.deg2rad(DECL)
    RLATR = np.deg2rad(RLAT)
    for II in range(ITER):  # DO 30....?
        if IPRT == 0:
            DELTIM = DELTME
            TIME = TERM + DELTIM
            # Read AMPLO, THERMC, WVN, PERIOD, PHI, TINT, TERM
        if TERM <= 98:
            ALT, GC, TEMP, WM, VNX0, VNY0, X, HK, TE, TI, PT, VLB,\
                DENN, DENO, BETAL0, PX, PZ, PN, GRADH, GRADHI, DEN, CHAP\
                = zprofl(
                ALT, GC, TEMP, WM, VNX0, VNY0, X, HK, TE, TI, PT,
                DENN, DENO, BETAL0, PX, PZ, PN, GRADH, GRADHI, DEN,
                RLATR, DECLR, FLUX, CONTV, TIME, TXMIN, TXMAX, T120,
                TEEMIN, TEEMAX, HNSTEP, COSI, SINI, QPO, ALITRN, 
                ALPHA, BETA0, II, J,\
            )
            
            A, B, C, QP, BETA, D, E, COLF, AMPGW = cofcal(
                A, B, C, QP, BETA, 1, HNSTEP, DELTIM, BCO, PX, PT, PN, PZ, GC,
                CHAP, ALT, SINI, COSI, COLF, AMPGW, DEN, BETAL0, ALPHA, J
            )
            V = tridia(HNSTEP, J, A, B, C, D, QP, BETA, VLB, VOB)
            for K in range(J):
                DEN[K] = V[K]
            DEN[J + 1] = DEN[J]
            if TIME >= TINT:
                WVNZI, WVNZR, PPK, PZK, PTK, PNK, PXK = facalg(
                    AMPL, AMPLO, GAMMA, WVN, HNSTEP, PERIOD, SINI, COSI, PHI, J, GC, HK,
                    DEN, TEMP, DENN, GRADH, VNX0, VNY0, TIME, X, Y, THERMC, TINT, PX, PZ, PT,
                )
                pdb.set_trace()
                if TIME == TINT:
                    sort(TINT)
                else: # (TIME > TINT)
                    cofcal(2)
                    tridia()
            TIME += DELTIM


# Define relevant functions

def zprofl(
        ALT, GC, TEMP, WM, VNX0, VNY0, X, HK, TE, TI, PT,
        DENN, DENO, BETAL0, PX, PZ, PN, GRADH, GRADHI, DEN,
        RLATR, DECLR, FLUX, CONTV, TIME, TXMIN, TXMAX, T120,
        TEEMIN, TEEMAX, HNSTEP, COSI, SINI, QPO, ALITRN, 
        ALPHA, BETA0, II, J,\
):

    CHI = np.arccos(np.sin(RLATR) * np.sin(DECLR) + np.cos(RLATR) \
            * np.cos(DECLR) * np.cos((TIME - 112.) * 15 / 57.296))
    TCOSC = (np.cos((CHI - .7854 + 0.20944 * np.sin(CHI + 0.7854)) * 0.5)) ** 2.5
    TEMPEX = TXMIN + (TXMAX - TXMIN) * TCOSC
    TEEX = TEEMIN + (TEEMAX - TEEMIN) * TCOSC
    TEX800 = (TEMPEX - 800.) ** 2
    SSS = 0.0291 * np.exp(-.5 * TEX800 / ((750. + (1.722E-4) * TEX800) ** 2))
    TEMSUB = TEMPEX - T120
    if II == 0:
        X[0] = 0.
        XSTEP = HNSTEP * COSI / (SINI * HCON)
        CHZ = np.arccos(np.sin(RLATR) * np.sin(DECLR) + np.cos(RLATR) \
                * np.cos(DECLR) * np.cos((19.112) * 15. / 57.296))
        TCOSZ = (np.cos((CHZ - 0.7854 + .20944 * np.sin(CHZ + 0.7854)) * .5)) ** 2.5
        TCFAC = TCOSC - TCOSZ
        PHOFLU = 6.8E8 * QPO
        ALT[0] = 120.
        GC[0] = 980.665 / ((1. + 120. / 6356.77) ** 2)
    CONT = 0.
    JJ = J + 1
    for K in np.arange(0, JJ):
        ALTEP = ALT[K]
        G = GC[K]
        TEMP[K] = TEMPEX - TEMSUB * np.exp((120. - ALTEP) * SSS)
        if ALT[K] == 200:
            IPRT = K
        if II == 0:
            if (ALTEP > 180):
                WM[K] = 25.106 - 7.9357 * np.arctan((ALTEP - 180.) / 140.)
            else:
                WM[K] = 20. - 5.0448 * np.arctan((ALTEP - 220.) / 25.)
            VNX0[K] = 0.
            VNY0[K] = 0.
            ALT[K + 1] = ALTEP + HNSTEP
            X[K + 1] = X[K] + XSTEP
            GC[K + 1] = 980.665 / ((1. + ALT[K + 1] / 6356.77) ** 2)
        HCOF1 = 831.44 * TEMP[K] / G
        HK[K] = HCOF1 / WM[K]
        HO = HCOF1 / 16.
        HN2 = HCOF1 / 28.
        TE[K] = TEMP[K] + (TEEX - TEMPEX) * np.exp(-80. / (ALTEP - 119.))
        if ALITRN <= ALT[K]:
            TI[K] = TE[K] + (TEMP[K] - TE[K]) * np.exp(1. - ALTEP / ALITRN)
        else:
            TI[K] = TEMP[K]
        PT[K] = TI[K] + TE[K]
        if K == 0:
            DENN[K] = DEN120
            DENO[0] = 7.6E10
            BETAL0[K] = BETA0
        else:
            PN[K - 1] = (PT[K] - PT[K - 1]) * HCON / HNSTEP
            BETAL0[K] = BETAL0[K - 1] * np.exp(-HNSTEP / HN2)
            GRADH[K - 1] = (HK[K] / HK[K - 1] - 1.) / HNSTEP
            DENN[K] = DENN[K - 1] * np.exp(-HNSTEP / HK[K] - GRADH[K - 1]) \
                        * TEMP[K - 1] / TEMP[K]
            DENO[K] = DENO[K - 1] * np.exp(-HNSTEP / HO) * TEMP[K - 1] / TEMP[K]
        COLFN = 1.565E15 * DENN[K] / (WM[K] ** 1.5)
        PX[K] = SINI / COLFN
        PZ[K] = VNX0[K] * COSI
        if (II != 0) and (DEN[K] >= CONT):
            CONT = DEN[K]  # DEN: ion density
    CHAP = produc(7.6E10, 4E11, TXMIN, 355., 28, 2.5, 0., 0., CHI, 1.3805E-16, \
            2.6512E-23, 3.6829E-23, GC[19], 6356.77, 120., HNSTEP, JJ, DENO, \
            17.32E-18, 14.1E-18, PHOFLU)
    if II == 0:
        for K in range(JJ):
            DEN[K] = np.sqrt(CHAP[K] / ALPHA)
    PN[JJ] = PN[J]
    GRADHI[JJ] = GRADH[J]
    # VLB is flux out of the ionosphere at height HL (order 1E8 cm-2 s-1)
    VLB = FLUX * CONT * (TCOSC - TCOSZ) / (TCFAC * CONTV)  
    print('VLB (ion flux out of the top %2.2E' % VLB)

    return  ALT, GC, TEMP, WM, VNX0, VNY0, X, HK, TE, TI, PT, VLB,\
            DENN, DENO, BETAL0, PX, PZ, PN, GRADH, GRADHI, DEN, CHAP
            
        
def cofcal(
        A, B, C, QP, BETA, IMPL, HNSTEP, DELTIM, BCO, PX, PT, PN, PZ, GC,
        CHAP, ALT, SINI, COSI, COLF, AMPGW, DEN, BETAL0, ALPHA, J,
):

    HNSTPG = HNSTEP / HCON
    HTRAN = 200.
    DELTIS = 3600 * DELTIM
    TC1 = BCO * PX[0] * PT[0]
    SC1 = GC[0] * PX[0] * BCO * PX[0] * PN[0] - PZ[0]
    for K in range(J):
        QP[K] = CHAP[K]  # ion production rate
        ALTEP = ALT[K]
        TC = TC1
        SC = SC1
        TC1 = BCO * PX[K + 1] * PT[K + 1]
        SC1 =                 BCO * PX[K + 1] * PN[K + 1] - PZ[K + 1] + GC[K + 1] * PX[K + 1]
        GRADTC = (TC1 - TC) / HNSTPG
        GRADSC = (SC1 - SC) / HNSTPG
        A[K] = TC * SINI
        B[K] = (GRADTC + SC) * SINI
        C[K] = GRADSC * SINI
        if IMPL != 1:
            DENK = COLF[K]
            QP[K] = QP[K] * AMPGW[K]
        else:
            DENK = DEN[K]
            BETAL = BETAL0[K] * AMPGW[K]
        if ALTEP <= HTRAN:
            ALPHN = ALPHA * DENK
            BETA[K] = ALPHN * BETAL / (BETAL + ALPHN)
        else:
            BETA[K] = BETAL
        C[K] = C[K] - 1. / DELTIS
    D = -A[J - 1]  # NOTE: -A[J] in the text
    E = - SC * SINI
    return A, B, C, QP, BETA, D, E, COLF, AMPGW


            
def facalg(
        AMPL, AMPLO, GAMMA, WVN, HNSTEP, PERIOD, SINI, COSI, PHI, J, GC, HK,
        DEN, TEMP, DENN, GRADH, VNX0, VNY0, TIME, X, Y, THERMC, TINT, PX, PZ, PT,
):
    """
    See page 110 of Clark thesis - calculates Kz
    # Translation from the outside world
    PSI: THERMK
    lambda_0: THERMC

    """ 
    IS = 0  # NOTE: not sure this is right
    WVNZTI = np.zeros(300)
    WVNZTR = np.zeros(300)
    WVNZI = np.zeros(300)
    WVNZR = np.zeros(300)
    TDELAY = np.zeros(300) 
    AMPGW = np.zeros(300) 
    PN = np.zeros(300) 

    if AMPL <= 0:
        AMPL = AMPLO
        G1 = GAMMA - 1.
        G2 = GAMMA / G1
        IS = 0
        WVN = WVN * HCON  
        HNSCM = HNSTEP / HCON
        DELTIM = PERIOD / 1200.
        SINISQ = SINI ** 2
        CSINI = COSI * SINI
        WVNSQ = WVN ** 2
        PHI = np.deg2rad(PHI)
        WVNX = WVN * np.cos(PHI)
        WVNXSQ = WVNX ** 2
        WVNY = WVN * np.sin(PHI)
        WVNYSQ = WVNSQ - WVNXSQ
        FREQ = .104718 / PERIOD  # 2 pi factor 
    AMLPHF = 0.
    IS += 1 
    JJ = J + 1
    for K in range(JJ):
        G = GC[K]
        HKK = HK[K] / HCON
        GHK = G * HKK
        DENK = DEN[K]
        TK = TEMP[K]
        HK12 = 0.5 / HKK
        HKSQ = 1. / (HKK * HKK)
        HKSQ4 = HKSQ / 4.
        COLLIF = COI * DENK
        PRESS = DENN[K] * GHK
        G2H = G2 * GRADH[K]
        G1H = 1. + G2H
        FRQ = FREQ - VNX0[K] * WVNX - VNY0[K] * WVNY
        FRTIM = FRQ * TIME * 3600.
        TDEL0 = WVNX / (FRQ * 3600.)
        PSIO = -Y * WVNY + FRTIM
        FRQSQ = FRQ ** 2
        WGH = FRQ / GHK
        F1 = H * COLLIF
        F2 = FRQ - F1
        F3 = FRQ - F1 * SINISQ
        PRESQ = 1. / np.sqrt(PRESS)
        THERMK = -H * THERMC * (TK ** 2.5) / ((TK + 245.4) * PRESS * FRQ)
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
                WVNZ = - np.sqrt(np.complex(-WVNSQ - HKSQ4 + W2GH \
                    / GAMMA + GKWH / G2, 0.))
                WVNT = WVNZ.copy()
            BB = C2 + THERMK * Z4
            CC = Z6 * (HKSQ4 + WVNSQ - W2GH) + G2H \
                * (HK12 / HKK - GKWH + H * WVNT / HKK) - GKWH + W2GH
            WVNT = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4.* CC * THERMK)) / THERM2)
            CC = Z6 * (HKSQ4 + WVNSQ - W2GH) + G2H \
                * (HK12 / HKK - GKWH + H * WVNT / HKK) - GKWH + W2GH
            WVNT = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4.* CC * THERMK)) / THERM2)
            Z5 = HKSQ4 - Z1 ** 2 - C1 * C3 - 2. * H * Z1 * WVNT
            BB = C2 + THERMK * (Z5 + HKSQ4)
            CC = Z6 * Z5 + G2H * (Z1 - HK12 + H * WVNT) / HKK + G1H* C1 * HKSQ / WGH - HKSQ + WGH * C3
            WVNZ = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4. * CC * THERMK)) / THERM2)
            WVNZTI[K] = np.imag(WVNT)
            WVNZTR[K] = np.real(WVNT)
        else:
            WVNZ = WVNZR[K] + H * WVNZI[K]
        Z5 = HKSQ4 - Z1 ** 2 - C1 * C3 - 2. * H * Z1 * WVNT
        BB = C2 + THERMK * (Z5 + HKSQ4)
        CC = Z6 * Z5 + G2H * (Z1 - HK12 + H * WVNZ) / HKK \
            + G1H * C1 * HKSQ / WGH - HKSQ + WGH * C3
        WVNZ = - np.sqrt((-BB + np.sqrt(BB ** 2 - 4. * CC * THERMK)) / THERM2)
        WVNZR[K] = np.real(WVNZ)
        WVNZI[K] = np.imag(WVNZ)
        PSIO = PSIO - WVNZ * HNSCM
        PSIK = PSIO - WVNX * X[K]
        Z2 = (WVNZ * WVNZ + HKSQ4) * THERMK
        Z3 = WVNZ - H * (HK12 + Z1)
        C4 = FRQSQ * G1 * PRESQ
        PPK = C4 * (Z3 * (G2 + Z2) + H * G1H / HKK)
        PZK = C4 * (C1 * GHK * (C2 + Z2) - FRQ)
        PTK = C4 * (H * C1 * G * G1H / FRQ + Z3)
        PNK = PPK - PTK
        PXK = (WVNX * GHK * PPK + COLLIF * CSINI * PZK) / F3
        TDELAY[K] = TDEL0 * X[K]
        FACT5 = AMPL * np.exp(H * PSIK)
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

    #      Kzz   P,    W,   T,   R,   U, 
    return WVNZI, WVNZR, PPK, PZK, PTK, PNK, PXK


def produc(
        DENOO, DENN2O, TNO, TNL, XRJ, XNJ, ALPHAO, ALPHAN, CHI, BC, XOM, XN2M,
         G, RE, HO, DELHN, N, DENO, ABO, ABN2, PHOFLU
):
    PROD = np.zeros(300)
    if CHI - 1.98 < 0:
        CHAPM1, CHAPM2, HNUK, ALP, ALPH = \
            chapmn(CHI, XOM, G, TNO, TNL, XRJ, XNJ, ALPHAO)
        B = DENOO * (TNL ** ALPH)
        DEPO1 = B * CHAPM1
        DEPO2 = B * CHAPM2
        HNUKO = HNUK
        ALPO = ALP
        CHAPM1, CHAPM2, HNUK, ALP, ALPH = \
            chapmn(CHI, XN2M, G, TNO, TNL, XRJ, XNJ, ALPHAN)
        C = DENN2O * (TNL ** ALPH)
        DEPN1 = C * CHAPM1
        DEPN2 = C * CHAPM2
        HNUKN = HNUK
        ALPN = ALP
        A = np.sin(CHI)
        ALTO = RE + HO
        for I in range(N):
            YI = I
            ALTD = YI * DELHN
            P = (ALTO + ALTD) * A
            if ((CHI - 1.5688) <= 0):
                DEPTHO = (DEPO1 * np.exp(-ALTD / HNUKO) + DEPO2 \
                    * np.exp(-ALPO * ALTD / HNUKO)) * P * ABO / HCON
                DEPTHN = (DEPN1 * np.exp(-ALTD / HNUKN) + DEPN2 \
                    * np.exp(-ALPN * ALTD / HNUKN)) * P * ABN2 / HCON
                SUM = DEPTHO + DEPTHN
                print('DEPTHO: %2.2f, DEPTHN: %2.2f' % (DEPTHO, DEPTHN))
                if (SUM - 150) <= 0:
                    PROD[I] = ABO * PHOFLU * np.exp(-SUM) * DENO[I]
                else:
                    PROD[I] = 0.
            else:
                if ((P - ALTO) > 0):
                    DEPTHO = (DEPO1 * np.exp(-ALTD / HNUKO) + DEPO2 \
                        * np.exp(-ALPO * ALTD / HNUKO)) * P * ABO / HCON
                    DEPTHN = (DEPN1 * np.exp(-ALTD / HNUKN) + DEPN2 \
                        * np.exp(-ALPN * ALTD / HNUKN)) * P * ABN2 / HCON
                    SUM = DEPTHO + DEPTHN
                    if (SUM - 150) <= 0:
                        PROD[I] = ABO * PHOFLU * np.exp(-SUM) * DENO[I]
                    else:
                        PROD[I] = 0.
                else:
                    PROD[I] = 0.
    else:
        for I in range(N):
            PROD[I] = 0.

    return PROD


def chapmn(CHI, XM, G, TNO, TNL, XRJ, XNJ, ALPHA):
    print('Calling Chapman function')
    P = 800.
    Q = 750.
    R = 1.722E-4

    TNU = TNO * (1. + XRJ * ((np.cos(CHI / 2.)) ** XNJ))
    S = TNU - P
    Y = S / (Q + R * (S ** 2))
    TS = 0.0291 * np.exp(-(Y ** 2) / 2.)
    HNU = (BC * TNU) / (XM * G)
    HNUK = HNU * HCON
    ALP = 1. + TS * HNUK
    ALPH = 1. + ALPHA + 1. / (TS * HNUK)
    if CHI - 0.0870 <= 0:
        CHAPM1 = (TNU ** (-ALPH)) * HNU
        CHAPM2 = (HNU / ALP) * (TNU ** (-ALPH - 1.)) * (TNU - TNL) * ALPH
    else:
        A = np.sin(CHI)
        X = 0.8170
        if  (CHI - 1.5688) <= 0:
            CHIX = np.arcsin(X * A)
        else:
            CHIX = 0.9560
        DCHI = (CHI - CHIX) / 200.
        TGIFI1 = 0.
        TGIFI2 = 0.
        GI11 = TNU ** (-ALPH) / A ** 2
        GI21 = ALPH * (TNU ** (-ALPH - 1.)) * (TNU - TNL) / A ** 2
        for I in range(200):
            YI = I
            CHI2 = CHI - (YI - 0.5) * DCHI
            CHI3 = CHI - YI * DCHI
            A2 = np.sin(CHI2)
            A3 = np.sin(CHI3)
            TNU2 = TNU * (1. + XRJ * ((np.cos(CHI2 / 2.)) ** XNJ))
            TNU3 = TNU * (1. + XRJ * ((np.cos(CHI3 / 2.)) ** XNJ))
            S2 = TNU2 - P
            S3 = TNU3 - P
            Y2 = S2 / (Q + R * (S2 ** 2))
            Y3 = S3 / (Q + R * (S3 ** 2))
            TS2 = 0.0291 * np.exp(-(Y2 ** 2) / 2.)
            TS3 = 0.0291 * np.exp(-(Y3 ** 2) / 2.)
            HNU2 = (BC * TNU2) / (XM * G)
            HNU3 = (BC * TNU3) / (XM * G)
            HNUK2 = HNU2 * HCON
            HNUK3 = HNU3 * HCON
            ALP2 = 1. + TS2 * HNUK2
            ALP3 = 1. + TS3 * HNUK3
            ALPH2 = ALPHA + 1. + 1. / (TS2 * HNUK2)
            ALPH3 = ALPHA + 1. + 1. / (TS3 * HNUK3)
            D2 = 6700. / HNUK2
            D3 = 6700. / HNUK3
            E2 = D2 * (A / A2 - 1.)
            E3 = D3 * (A / A3 - 1.)
            GI12 = (TNU2 ** (-ALPH2)) * np.exp(-E2) / A2 ** 2
            GI22 = ALPH2 * (TNU2 ** (-ALPH2 - 1.)) * (TNU2 - TNL) \
                * np.exp(-ALP2 * E2) / A2 ** 2
            GI13 = (TNU3 ** (-ALPH3)) * np.exp(-E3) / A3 ** 2
            GI23 = ALPH3 * (TNU3 ** (-ALPH3 - 1.)) * (TNU3 - TNL) \
                * np.exp(-ALP3 * E3) / A3 ** 2
            GIFI1 = (GI11 + 4. * GI12 + GI13) * DCHI / 6.
            GIFI2 = (GI21 + 4. * GI22 + GI23) * DCHI / 6.
            TGIFI1 = TGIFI1 + GIFI1
            TGIFI2 = TGIFI2 + GIFI2
            GI11 = GI13
            GI21 = GI23
        CHAPM1 = TGIFI1
        CHAPM2 = TGIFI2
    return CHAPM1, CHAPM2, HNUK, ALP, ALPH


def tridia(HNSTEP, J, A, B, C, D, QP, BETA, VLB, VOB):
    """
    QP - ion production rate
    BETA - ionization attachment coefficient
    Why is BETA nan? Why is D 0.0?
    """
    P = np.zeros(300)
    Q = np.zeros(300)
    R = np.zeros(300)
    S = np.zeros(300)
    U = np.zeros(300)
    V = np.zeros(300)

    L = J
    DELX = HNSTEP / HCON
    DELXSQ = DELX ** 2
    TODELX = 2. * DELX
    for I  in range(L):
        AC = A[I] / DELXSQ
        BC = B[I] / TODELX
        P[I] = AC - BC
        Q[I] = C[I] - BETA[I] - 2. * AC
        R[I] = AC + BC
        if I == L:
            P[I] = P[I] + R[I]
            Q[I] = Q[I] - ((TODELX * E * R[I]) / D)
            R[I] = (TODELX * R[I]) / D
    GLB = -QP[L] - R[L] * VLB
    GOB = -QP[0] - P[0] * VOB

    # Calculation of off-diagonal elements of U matrix and elements of S
    for I in range(L):
        if (I - 1) == 0:
            UNUM = R[I]
            DEM = Q[I]
            SNUM = GOB
        elif (I - L) == 0:
            UNUM = 0.
            DEM = Q[I] - P[I] * U[I - 1]
            SNUM = GLB - P[I] * S[I - 1]
        else:
            UNUM = R[I]
            DEM = Q[I] - P[I] * U[I - 1]
            SNUM = -QP[I] - P[I] * S[I - 1]
        S[I] = SNUM / DEM
        U[I] = UNUM / DEM

    # Solution for the canonical matrix equation for V[N]
    V[L] = S[L]
    L = L - 1
    N = L
    for I in range(L):
        V[N] = S[N] - U[N] * V[N + 1]
        N = N - 1
    return V


### End of function definitions ###
   

def first_try():
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
