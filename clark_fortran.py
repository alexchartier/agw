import numpy as np
import pdb 
import matplotlib.pyplot as plt

"""
AGW/TID simulation based Clark (1970) thesis
Inputs: Period, horizontal wavelength, direction, neutral wind disturbance 
        amplitude at lowest height, climatological inputs
Outputs: 3D estimate of:
            1. neutral velocity, temperature and density perturbations
            2. ion velocity, temperature (i and e) and density perturbations
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
def get_clark_Kz():

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
    TINT = 12.  # Time of start of GW in hr (must be > 12 or the last TERM read, and even multiple of DELTME)
    TERM = 24.  # Time of GW termination in hr (> TINT and even multiple of DELTME)
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
    #for II in range(ITER):  # DO 30....
    II = 0
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
                AMPL, AMPLO, GAMMA, WVN, HNSTEP, PERIOD, SINI, COSI, 
                PHI, J, GC, HK, DEN, TEMP, DENN, GRADH, VNX0, VNY0, 
                TIME, X, Y, THERMC, TINT, PX, PZ, PT, ALT,
            )
            """
            if TIME == TINT:
                sort(TINT)
            else: # (TIME > TINT)
                cofcal(2)
                tridia()
            """
        TIME += DELTIM
    return WVNZI, WVNZR, ALT


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
            7.32E-18, 14.1E-18, PHOFLU, ALT)
    if II == 0:
        for K in range(JJ):
            DEN[K] = np.sqrt(CHAP[K] / ALPHA)
    pdb.set_trace()
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
        SC1 = BCO * PX[K + 1] * PN[K + 1] - PZ[K + 1] + GC[K + 1] * PX[K + 1]
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
        AMPL, AMPLO, GAMMA, WVN, HNSTEP, PERIOD, SINI, COSI, 
        PHI, J, GC, HK, DEN, TEMP, DENN, GRADH, VNX0, VNY0, 
        TIME, X, Y, THERMC, TINT, PX, PZ, PT, ALT,
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
        print(
            'ALT: %i, COLLIF: %2.2E ' \
            % (ALT[K], COLLIF)
        )
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
            CC = Z6 * Z5 + G2H * (Z1 - HK12 + H * WVNT) / HKK + \
                    G1H* C1 * HKSQ / WGH - HKSQ + WGH * C3
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
        G, RE, HO, DELHN, N, DENO, ABO, ABN2, PHOFLU, ALT,
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
                pdb.set_trace()
                print('ALT: %i, DEPTHO: %2.2f, DEPTHN: %2.2f' \
                    % (ALT[I], DEPTHO, DEPTHN))
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
   
if __name__ == '__main__':
    get_clark_Kz()
