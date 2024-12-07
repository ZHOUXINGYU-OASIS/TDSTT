import numpy as np
import math
import scipy
from scipy.integrate import solve_ivp
from Module_STT import STM_pred, STT2_pred

def derivative_matrix_distinct(K, dK):
    """Calculate the derivatives of the eigenvalues and eigenvectors of the matrix K"""
    DIM = 6
    eigenvalue, featurevector = np.linalg.eig(K)
    featurevector = featurevector.T
    Coeff = np.zeros([DIM, DIM])
    for i in range(DIM):
        ni = featurevector[i]  # i-th eigenvector
        lami = eigenvalue[i]  # i-th eigenvalue
        for j in range(DIM):
            nj = featurevector[j]  # j-th eigenvector
            lamj = eigenvalue[j]  # i-th eigenvalue
            if i == j:
                continue
            else:
                Coeff[i, j] = 1 / (lami - lamj) * np.dot(dK @ ni, nj)
    return featurevector, Coeff

def CRTBP_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model"""
    r1 = math.sqrt((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    r2 = math.sqrt((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    m1 = 1 - mu
    m2 = mu
    dydt = np.array([
        y[3],
        y[4],
        y[5],
        y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3),
        y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3),
        -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    ])
    return dydt

def CRTBP_STM_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM)"""
    x = y[:6]
    STM = y[6:].reshape(6, 6)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    A = cal_1st_tensor(x, mu)
    dSTM = np.matmul(A, STM).reshape(36)
    dy = np.concatenate((dxdt, dSTM))
    return dy

def CRTBP_STT2_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM and STT)"""
    x = y[:6]
    STM = y[6:42].reshape(6, 6)
    STT = y[42:].reshape(6, 6, 6)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    N1 = cal_1st_tensor(x, mu)
    dSTM = np.matmul(N1, STM).reshape(36)
    """STT"""
    N2 = cal_2nd_tensor(x, mu)
    dSTT = np.zeros([6, 6, 6])
    for i in range(6):
        for a in range(6):
            for b in range(6):
                for alpha in range(6):
                    dSTT[i, a, b] = dSTT[i, a, b] + N1[i, alpha] * STT[alpha, a, b]
                    for beta in range(6):
                        dSTT[i, a, b] = dSTT[i, a, b] + N2[i, alpha, beta] * STM[alpha, a] * STM[beta, b]
    dSTT = dSTT.reshape(6 ** 3)
    dy = np.concatenate((dxdt, dSTM, dSTT))
    return dy

def CRTBP_STT3_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM, STT, and STT3)"""
    DIM = 6
    x = y[:DIM]
    STM = y[DIM:(DIM ** 2 + DIM)].reshape(DIM, DIM)
    STT = y[(DIM ** 2 + DIM):(DIM ** 3 + DIM ** 2 + DIM)].reshape(DIM, DIM, DIM)
    STT3 = y[(DIM ** 3 + DIM ** 2 + DIM):(DIM ** 4 + DIM ** 3 + DIM ** 2 + DIM)].reshape(DIM, DIM, DIM, DIM)
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    N1 = cal_1st_tensor(x, mu)
    dSTM = np.matmul(N1, STM).reshape(DIM ** 2)
    """STT"""
    N2 = cal_2nd_tensor(x, mu)
    dSTT = np.zeros([DIM, DIM, DIM])
    for i in range(DIM):
        for a in range(DIM):
            for b in range(DIM):
                for alpha in range(DIM):
                    dSTT[i, a, b] = dSTT[i, a, b] + N1[i, alpha] * STT[alpha, a, b]
                    for beta in range(DIM):
                        dSTT[i, a, b] = dSTT[i, a, b] + N2[i, alpha, beta] * STM[alpha, a] * STM[beta, b]
    dSTT = dSTT.reshape(DIM ** 3)
    """STT3"""
    N3 = cal_3rd_tensor(x, mu)
    dSTT3 = np.zeros([DIM, DIM, DIM, DIM])
    for i in range(DIM):
        for a in range(DIM):
            for b in range(DIM):
                for c in range(DIM):
                    for alpha in range(DIM):  # 1
                        dSTT3[i, a, b, c] = dSTT3[i, a, b, c] + N1[i, alpha] * STT3[alpha, a, b, c]
                    for alpha in range(DIM):  # 2
                        for beta in range(DIM):
                            dSTT3[i, a, b, c] = dSTT3[i, a, b, c] + N2[i, alpha, beta] * (STM[alpha, a] * STT[beta, b, c] + STT[alpha, a, b] * STM[beta, c] + STT[alpha, a, c] * STM[beta, b])
                    for alpha in range(DIM):  # 3
                        for beta in range(DIM):
                            for gamma in range(DIM):
                                dSTT3[i, a, b, c] = dSTT3[i, a, b, c] + N3[i, alpha, beta, gamma] * STM[alpha, a] * STM[beta, b] * STM[gamma, c]
    dSTT3 = dSTT3.reshape(DIM ** 4)
    """Output"""
    dy = np.concatenate((dxdt, dSTM, dSTT, dSTT3))
    return dy

def cal_1st_tensor(x, mu):
    """the first-order tensor of the CRTBP dynamcis"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    daxdrx = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) + 1
    daxdry = (3 * mu * ry * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdrz = (3 * mu * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdvx = 0
    daxdvy = 2
    daxdvz = 0
    daydrx = (3 * mu * ry * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * ry * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    daydry = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * ry ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + 1
    daydrz = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daydvx = -2
    daydvy = 0
    daydvz = 0
    dazdrx = (3 * mu * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    dazdry = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    dazdrz = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    dazdvx = 0
    dazdvy = 0
    dazdvz = 0
    """Jacobi matrix"""
    A = np.zeros([6, 6])
    A[:3, 3:] = np.eye(3)
    A[3:, :] = np.array([
        [daxdrx, daxdry, daxdrz, daxdvx, daxdvy, daxdvz],
        [daydrx, daydry, daydrz, daydvx, daydvy, daydvz],
        [dazdrx, dazdry, dazdrz, dazdvx, dazdvy, dazdvz],
    ])
    return A

def cal_2nd_tensor(x, mu):
    """the second-order tensor of the CRTBP dynamcis"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    A = np.zeros([6, 6, 6])
    """elements of A"""
    daxdrxrx = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (3 * mu * (2 * mu + 2 * rx - 2)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxvx = 0
    daxdrxvy = 0
    daxdrxvz = 0
    daxdryrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdryry = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * ry ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdryrz = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdryvx = 0
    daxdryvy = 0
    daxdryvz = 0
    daxdrzrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrzry = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzrz = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * rz ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * rz ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzvx = 0
    daxdrzvy = 0
    daxdrzvz = 0
    daxdvxrx = 0
    daxdvxry = 0
    daxdvxrz = 0
    daxdvxvx = 0
    daxdvxvy = 0
    daxdvxvz = 0
    daxdvyrx = 0
    daxdvyry = 0
    daxdvyrz = 0
    daxdvyvx = 0
    daxdvyvy = 0
    daxdvyvz = 0
    daxdvzrx = 0
    daxdvzry = 0
    daxdvzrz = 0
    daxdvzvx = 0
    daxdvzvy = 0
    daxdvzvz = 0
    daydrxrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxry = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxrz = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxvx = 0
    daydrxvy = 0
    daydrxvz = 0
    daydryrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydryry = (9 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydryrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydryvx = 0
    daydryvy = 0
    daydryvz = 0
    daydrzrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrzry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzvx = 0
    daydrzvy = 0
    daydrzvz = 0
    daydvxrx = 0
    daydvxry = 0
    daydvxrz = 0
    daydvxvx = 0
    daydvxvy = 0
    daydvxvz = 0
    daydvyrx = 0
    daydvyry = 0
    daydvyrz = 0
    daydvyvx = 0
    daydvyvy = 0
    daydvyvz = 0
    daydvzrx = 0
    daydvzry = 0
    daydvzrz = 0
    daydvzvx = 0
    daydvzvy = 0
    daydvzvz = 0
    dazdrxrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxry = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxrz = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxvx = 0
    dazdrxvy = 0
    dazdrxvz = 0
    dazdryrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdryry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdryrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdryvx = 0
    dazdryvy = 0
    dazdryvz = 0
    dazdrzrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrzry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzrz = (9 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * rz ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzvx = 0
    dazdrzvy = 0
    dazdrzvz = 0
    dazdvxrx = 0
    dazdvxry = 0
    dazdvxrz = 0
    dazdvxvx = 0
    dazdvxvy = 0
    dazdvxvz = 0
    dazdvyrx = 0
    dazdvyry = 0
    dazdvyrz = 0
    dazdvyvx = 0
    dazdvyvy = 0
    dazdvyvz = 0
    dazdvzrx = 0
    dazdvzry = 0
    dazdvzrz = 0
    dazdvzvx = 0
    dazdvzvy = 0
    dazdvzvz = 0
    A[3] = np.array([
        [daxdrxrx, daxdrxry, daxdrxrz, daxdrxvx, daxdrxvy, daxdrxvz],
        [daxdryrx, daxdryry, daxdryrz, daxdryvx, daxdryvy, daxdryvz],
        [daxdrzrx, daxdrzry, daxdrzrz, daxdrzvx, daxdrzvy, daxdrzvz],
        [daxdvxrx, daxdvxry, daxdvxrz, daxdvxvx, daxdvxvy, daxdvxvz],
        [daxdvyrx, daxdvyry, daxdvyrz, daxdvyvx, daxdvyvy, daxdvyvz],
        [daxdvzrx, daxdvzry, daxdvzrz, daxdvzvx, daxdvzvy, daxdvzvz],
    ])
    A[4] = np.array([
        [daydrxrx, daydrxry, daydrxrz, daydrxvx, daydrxvy, daydrxvz],
        [daydryrx, daydryry, daydryrz, daydryvx, daydryvy, daydryvz],
        [daydrzrx, daydrzry, daydrzrz, daydrzvx, daydrzvy, daydrzvz],
        [daydvxrx, daydvxry, daydvxrz, daydvxvx, daydvxvy, daydvxvz],
        [daydvyrx, daydvyry, daydvyrz, daydvyvx, daydvyvy, daydvyvz],
        [daydvzrx, daydvzry, daydvzrz, daydvzvx, daydvzvy, daydvzvz],
    ])
    A[5] = np.array([
        [dazdrxrx, dazdrxry, dazdrxrz, dazdrxvx, dazdrxvy, dazdrxvz],
        [dazdryrx, dazdryry, dazdryrz, dazdryvx, dazdryvy, dazdryvz],
        [dazdrzrx, dazdrzry, dazdrzrz, dazdrzvx, dazdrzvy, dazdrzvz],
        [dazdvxrx, dazdvxry, dazdvxrz, dazdvxvx, dazdvxvy, dazdvxvz],
        [dazdvyrx, dazdvyry, dazdvyrz, dazdvyvx, dazdvyvy, dazdvyvz],
        [dazdvzrx, dazdvzry, dazdvzrz, dazdvzvx, dazdvzvy, dazdvzvz],
    ])
    return A

def cal_3rd_tensor(x, mu):
    """the third-order tensor of the CRTBP dynamcis"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    A = np.zeros([6, 6, 6, 6])
    """elements of A"""
    daxdrxrxrx = (9*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (9*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (45*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (45*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*(mu + rx - 1)*(8*mu + 8*rx - 8))/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(8*mu + 8*rx))/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*(mu + rx - 1)*(2*mu + 2*rx - 2)**3)/(8*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**3)/(8*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrxry = (15*ry*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrxrz = (15*rz*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrxvx = 0
    daxdrxrxvy = 0
    daxdrxrxvz = 0
    daxdrxryrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxryry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*ry**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxryrz = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxryvx = 0
    daxdrxryvy = 0
    daxdrxryvz = 0
    daxdrxrzrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrzry = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrzrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*rz**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrxrzvx = 0
    daxdrxrzvy = 0
    daxdrxrzvz = 0
    daxdrxvxrx = 0
    daxdrxvxry = 0
    daxdrxvxrz = 0
    daxdrxvxvx = 0
    daxdrxvxvy = 0
    daxdrxvxvz = 0
    daxdrxvyrx = 0
    daxdrxvyry = 0
    daxdrxvyrz = 0
    daxdrxvyvx = 0
    daxdrxvyvy = 0
    daxdrxvyvz = 0
    daxdrxvzrx = 0
    daxdrxvzry = 0
    daxdrxvzrz = 0
    daxdrxvzvx = 0
    daxdrxvzvy = 0
    daxdrxvzvz = 0
    daxdryrxrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdryrxry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*ry**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdryrxrz = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdryrxvx = 0
    daxdryrxvy = 0
    daxdryrxvz = 0
    daxdryryrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*ry**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdryryry = (45*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (45*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry**3*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) + (105*mu*ry**3*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)
    daxdryryrz = (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdryryvx = 0
    daxdryryvy = 0
    daxdryryvz = 0
    daxdryrzrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdryrzry = (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdryrzrz = (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**2*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**2*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdryrzvx = 0
    daxdryrzvy = 0
    daxdryrzvz = 0
    daxdryvxrx = 0
    daxdryvxry = 0
    daxdryvxrz = 0
    daxdryvxvx = 0
    daxdryvxvy = 0
    daxdryvxvz = 0
    daxdryvyrx = 0
    daxdryvyry = 0
    daxdryvyrz = 0
    daxdryvyvx = 0
    daxdryvyvy = 0
    daxdryvyvz = 0
    daxdryvzrx = 0
    daxdryvzry = 0
    daxdryvzrz = 0
    daxdryvzvx = 0
    daxdryvzvy = 0
    daxdryvzvz = 0
    daxdrzrxrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(2*mu + 2*rx - 2))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz*(mu + rx - 1)*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrzrxry = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrzrxrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*rz**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrzrxvx = 0
    daxdrzrxvy = 0
    daxdrzrxvz = 0
    daxdrzryrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*ry*rz*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry*rz*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrzryry = (15*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdrzryrz = (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**2*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**2*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdrzryvx = 0
    daxdrzryvy = 0
    daxdrzryvz = 0
    daxdrzrzrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (105*rz**2*(mu + rx)*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(mu + rx - 1)*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daxdrzrzry = (15*ry*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**2*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**2*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daxdrzrzrz = (45*rz*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (45*mu*rz*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (105*rz**3*(mu + rx)*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) + (105*mu*rz**3*(mu + rx - 1))/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)
    daxdrzrzvx = 0
    daxdrzrzvy = 0
    daxdrzrzvz = 0
    daxdrzvxrx = 0
    daxdrzvxry = 0
    daxdrzvxrz = 0
    daxdrzvxvx = 0
    daxdrzvxvy = 0
    daxdrzvxvz = 0
    daxdrzvyrx = 0
    daxdrzvyry = 0
    daxdrzvyrz = 0
    daxdrzvyvx = 0
    daxdrzvyvy = 0
    daxdrzvyvz = 0
    daxdrzvzrx = 0
    daxdrzvzry = 0
    daxdrzvzrz = 0
    daxdrzvzvx = 0
    daxdrzvzvy = 0
    daxdrzvzvz = 0
    daxdvxrxrx = 0
    daxdvxrxry = 0
    daxdvxrxrz = 0
    daxdvxrxvx = 0
    daxdvxrxvy = 0
    daxdvxrxvz = 0
    daxdvxryrx = 0
    daxdvxryry = 0
    daxdvxryrz = 0
    daxdvxryvx = 0
    daxdvxryvy = 0
    daxdvxryvz = 0
    daxdvxrzrx = 0
    daxdvxrzry = 0
    daxdvxrzrz = 0
    daxdvxrzvx = 0
    daxdvxrzvy = 0
    daxdvxrzvz = 0
    daxdvxvxrx = 0
    daxdvxvxry = 0
    daxdvxvxrz = 0
    daxdvxvxvx = 0
    daxdvxvxvy = 0
    daxdvxvxvz = 0
    daxdvxvyrx = 0
    daxdvxvyry = 0
    daxdvxvyrz = 0
    daxdvxvyvx = 0
    daxdvxvyvy = 0
    daxdvxvyvz = 0
    daxdvxvzrx = 0
    daxdvxvzry = 0
    daxdvxvzrz = 0
    daxdvxvzvx = 0
    daxdvxvzvy = 0
    daxdvxvzvz = 0
    daxdvyrxrx = 0
    daxdvyrxry = 0
    daxdvyrxrz = 0
    daxdvyrxvx = 0
    daxdvyrxvy = 0
    daxdvyrxvz = 0
    daxdvyryrx = 0
    daxdvyryry = 0
    daxdvyryrz = 0
    daxdvyryvx = 0
    daxdvyryvy = 0
    daxdvyryvz = 0
    daxdvyrzrx = 0
    daxdvyrzry = 0
    daxdvyrzrz = 0
    daxdvyrzvx = 0
    daxdvyrzvy = 0
    daxdvyrzvz = 0
    daxdvyvxrx = 0
    daxdvyvxry = 0
    daxdvyvxrz = 0
    daxdvyvxvx = 0
    daxdvyvxvy = 0
    daxdvyvxvz = 0
    daxdvyvyrx = 0
    daxdvyvyry = 0
    daxdvyvyrz = 0
    daxdvyvyvx = 0
    daxdvyvyvy = 0
    daxdvyvyvz = 0
    daxdvyvzrx = 0
    daxdvyvzry = 0
    daxdvyvzrz = 0
    daxdvyvzvx = 0
    daxdvyvzvy = 0
    daxdvyvzvz = 0
    daxdvzrxrx = 0
    daxdvzrxry = 0
    daxdvzrxrz = 0
    daxdvzrxvx = 0
    daxdvzrxvy = 0
    daxdvzrxvz = 0
    daxdvzryrx = 0
    daxdvzryry = 0
    daxdvzryrz = 0
    daxdvzryvx = 0
    daxdvzryvy = 0
    daxdvzryvz = 0
    daxdvzrzrx = 0
    daxdvzrzry = 0
    daxdvzrzrz = 0
    daxdvzrzvx = 0
    daxdvzrzvy = 0
    daxdvzrzvz = 0
    daxdvzvxrx = 0
    daxdvzvxry = 0
    daxdvzvxrz = 0
    daxdvzvxvx = 0
    daxdvzvxvy = 0
    daxdvzvxvz = 0
    daxdvzvyrx = 0
    daxdvzvyry = 0
    daxdvzvyrz = 0
    daxdvzvyvx = 0
    daxdvzvyvy = 0
    daxdvzvyvz = 0
    daxdvzvzrx = 0
    daxdvzvzry = 0
    daxdvzvzrz = 0
    daxdvzvzvx = 0
    daxdvzvzvy = 0
    daxdvzvzvz = 0
    daydrxrxrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(8*mu + 8*rx - 8))/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*ry*(mu - 1)*(8*mu + 8*rx))/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*(2*mu + 2*rx - 2)**3)/(8*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*(mu - 1)*(2*mu + 2*rx)**3)/(8*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxrxry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daydrxrxrz = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxrxvx = 0
    daydrxrxvy = 0
    daydrxrxvz = 0
    daydrxryrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daydrxryry = (45*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxryrz = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxryvx = 0
    daydrxryvy = 0
    daydrxryvz = 0
    daydrxrzrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxrzry = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxrzrz = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrxrzvx = 0
    daydrxrzvy = 0
    daydrxrzvz = 0
    daydrxvxrx = 0
    daydrxvxry = 0
    daydrxvxrz = 0
    daydrxvxvx = 0
    daydrxvxvy = 0
    daydrxvxvz = 0
    daydrxvyrx = 0
    daydrxvyry = 0
    daydrxvyrz = 0
    daydrxvyvx = 0
    daydrxvyvy = 0
    daydrxvyvz = 0
    daydrxvzrx = 0
    daydrxvzry = 0
    daydrxvzrz = 0
    daydrxvzvx = 0
    daydrxvzvy = 0
    daydrxvzvz = 0
    daydryrxrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*ry**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    daydryrxry = (45*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydryrxrz = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydryrxvx = 0
    daydryrxvy = 0
    daydryrxvz = 0
    daydryryrx = (45*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydryryry = (9*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (9*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (90*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**4)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) + (90*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*ry**4*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daydryryrz = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**3*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**3*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    daydryryvx = 0
    daydryryvy = 0
    daydryryvz = 0
    daydryrzrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydryrzry = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**3*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**3*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    daydryrzrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daydryrzvx = 0
    daydryrzvy = 0
    daydryrzvz = 0
    daydryvxrx = 0
    daydryvxry = 0
    daydryvxrz = 0
    daydryvxvx = 0
    daydryvxvy = 0
    daydryvxvz = 0
    daydryvyrx = 0
    daydryvyry = 0
    daydryvyrz = 0
    daydryvyvx = 0
    daydryvyvy = 0
    daydryvyvz = 0
    daydryvzrx = 0
    daydryvzry = 0
    daydryvzrz = 0
    daydryvzvx = 0
    daydryvzvy = 0
    daydryvzvz = 0
    daydrzrxrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrzrxry = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrzrxrz = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrzrxvx = 0
    daydrzrxvy = 0
    daydrzrxvz = 0
    daydrzryrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrzryry = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**3*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**3*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    daydrzryrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daydrzryvx = 0
    daydrzryvy = 0
    daydrzryvz = 0
    daydrzrzrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    daydrzrzry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    daydrzrzrz = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**3)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    daydrzrzvx = 0
    daydrzrzvy = 0
    daydrzrzvz = 0
    daydrzvxrx = 0
    daydrzvxry = 0
    daydrzvxrz = 0
    daydrzvxvx = 0
    daydrzvxvy = 0
    daydrzvxvz = 0
    daydrzvyrx = 0
    daydrzvyry = 0
    daydrzvyrz = 0
    daydrzvyvx = 0
    daydrzvyvy = 0
    daydrzvyvz = 0
    daydrzvzrx = 0
    daydrzvzry = 0
    daydrzvzrz = 0
    daydrzvzvx = 0
    daydrzvzvy = 0
    daydrzvzvz = 0
    daydvxrxrx = 0
    daydvxrxry = 0
    daydvxrxrz = 0
    daydvxrxvx = 0
    daydvxrxvy = 0
    daydvxrxvz = 0
    daydvxryrx = 0
    daydvxryry = 0
    daydvxryrz = 0
    daydvxryvx = 0
    daydvxryvy = 0
    daydvxryvz = 0
    daydvxrzrx = 0
    daydvxrzry = 0
    daydvxrzrz = 0
    daydvxrzvx = 0
    daydvxrzvy = 0
    daydvxrzvz = 0
    daydvxvxrx = 0
    daydvxvxry = 0
    daydvxvxrz = 0
    daydvxvxvx = 0
    daydvxvxvy = 0
    daydvxvxvz = 0
    daydvxvyrx = 0
    daydvxvyry = 0
    daydvxvyrz = 0
    daydvxvyvx = 0
    daydvxvyvy = 0
    daydvxvyvz = 0
    daydvxvzrx = 0
    daydvxvzry = 0
    daydvxvzrz = 0
    daydvxvzvx = 0
    daydvxvzvy = 0
    daydvxvzvz = 0
    daydvyrxrx = 0
    daydvyrxry = 0
    daydvyrxrz = 0
    daydvyrxvx = 0
    daydvyrxvy = 0
    daydvyrxvz = 0
    daydvyryrx = 0
    daydvyryry = 0
    daydvyryrz = 0
    daydvyryvx = 0
    daydvyryvy = 0
    daydvyryvz = 0
    daydvyrzrx = 0
    daydvyrzry = 0
    daydvyrzrz = 0
    daydvyrzvx = 0
    daydvyrzvy = 0
    daydvyrzvz = 0
    daydvyvxrx = 0
    daydvyvxry = 0
    daydvyvxrz = 0
    daydvyvxvx = 0
    daydvyvxvy = 0
    daydvyvxvz = 0
    daydvyvyrx = 0
    daydvyvyry = 0
    daydvyvyrz = 0
    daydvyvyvx = 0
    daydvyvyvy = 0
    daydvyvyvz = 0
    daydvyvzrx = 0
    daydvyvzry = 0
    daydvyvzrz = 0
    daydvyvzvx = 0
    daydvyvzvy = 0
    daydvyvzvz = 0
    daydvzrxrx = 0
    daydvzrxry = 0
    daydvzrxrz = 0
    daydvzrxvx = 0
    daydvzrxvy = 0
    daydvzrxvz = 0
    daydvzryrx = 0
    daydvzryry = 0
    daydvzryrz = 0
    daydvzryvx = 0
    daydvzryvy = 0
    daydvzryvz = 0
    daydvzrzrx = 0
    daydvzrzry = 0
    daydvzrzrz = 0
    daydvzrzvx = 0
    daydvzrzvy = 0
    daydvzrzvz = 0
    daydvzvxrx = 0
    daydvzvxry = 0
    daydvzvxrz = 0
    daydvzvxvx = 0
    daydvzvxvy = 0
    daydvzvxvz = 0
    daydvzvyrx = 0
    daydvzvyry = 0
    daydvzvyrz = 0
    daydvzvyvx = 0
    daydvzvyvy = 0
    daydvzvyvz = 0
    daydvzvzrx = 0
    daydvzvzry = 0
    daydvzvzrz = 0
    daydvzvzvx = 0
    daydvzvzvy = 0
    daydvzvzvz = 0
    dazdrxrxrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(8*mu + 8*rx - 8))/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*rz*(mu - 1)*(8*mu + 8*rx))/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*rz*(2*mu + 2*rx - 2)**3)/(8*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*rz*(mu - 1)*(2*mu + 2*rx)**3)/(8*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrxry = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrxrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrxvx = 0
    dazdrxrxvy = 0
    dazdrxrxvz = 0
    dazdrxryrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxryry = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxryrz = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxryvx = 0
    dazdrxryvy = 0
    dazdrxryvz = 0
    dazdrxrzrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrzry = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrzrz = (45*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*rz**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*rz**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrxrzvx = 0
    dazdrxrzvy = 0
    dazdrxrzvz = 0
    dazdrxvxrx = 0
    dazdrxvxry = 0
    dazdrxvxrz = 0
    dazdrxvxvx = 0
    dazdrxvxvy = 0
    dazdrxvxvz = 0
    dazdrxvyrx = 0
    dazdrxvyry = 0
    dazdrxvyrz = 0
    dazdrxvyvx = 0
    dazdrxvyvy = 0
    dazdrxvyvz = 0
    dazdrxvzrx = 0
    dazdrxvzry = 0
    dazdrxvzrz = 0
    dazdrxvzvx = 0
    dazdrxvzvy = 0
    dazdrxvzvz = 0
    dazdryrxrx = (15*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (15*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdryrxry = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdryrxrz = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdryrxvx = 0
    dazdryrxvy = 0
    dazdryrxvz = 0
    dazdryryrx = (15*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry**2*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry**2*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdryryry = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**3*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**3*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    dazdryryrz = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    dazdryryvx = 0
    dazdryryvy = 0
    dazdryryvz = 0
    dazdryrzrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdryrzry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    dazdryrzrz = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**3)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    dazdryrzvx = 0
    dazdryrzvy = 0
    dazdryrzvz = 0
    dazdryvxrx = 0
    dazdryvxry = 0
    dazdryvxrz = 0
    dazdryvxvx = 0
    dazdryvxvy = 0
    dazdryvxvz = 0
    dazdryvyrx = 0
    dazdryvyry = 0
    dazdryvyrz = 0
    dazdryvyvx = 0
    dazdryvyvy = 0
    dazdryvyvz = 0
    dazdryvzrx = 0
    dazdryvzry = 0
    dazdryvzrz = 0
    dazdryvzvx = 0
    dazdryvzvy = 0
    dazdryvzvz = 0
    dazdrzrxrx = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (15*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz**2*(mu - 1)*(2*mu + 2*rx)**2)/(4*((mu + rx)**2 + ry**2 + rz**2)**(9/2)) + (105*mu*rz**2*(2*mu + 2*rx - 2)**2)/(4*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2))
    dazdrzrxry = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrzrxrz = (45*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*rz**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*rz**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrzrxvx = 0
    dazdrzrxvy = 0
    dazdrzrxvz = 0
    dazdrzryrx = (15*ry*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (15*mu*ry*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*ry*rz**2*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*ry*rz**2*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrzryry = (3*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (15*mu*ry**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) - (15*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (15*ry**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (15*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry**2*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry**2*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    dazdrzryrz = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**3)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    dazdrzryvx = 0
    dazdrzryvy = 0
    dazdrzryvz = 0
    dazdrzrzrx = (45*rz*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(7/2)) - (45*mu*rz*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)) + (105*mu*rz**3*(2*mu + 2*rx - 2))/(2*((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2)) - (105*rz**3*(mu - 1)*(2*mu + 2*rx))/(2*((mu + rx)**2 + ry**2 + rz**2)**(9/2))
    dazdrzrzry = (45*ry*rz*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) + (105*mu*ry*rz**3)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) - (105*ry*rz**3*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2) - (45*mu*ry*rz)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2)
    dazdrzrzrz = (9*mu)/((mu + rx - 1)**2 + ry**2 + rz**2)**(5/2) - (9*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(5/2) - (90*mu*rz**2)/((mu + rx - 1)**2 + ry**2 + rz**2)**(7/2) + (105*mu*rz**4)/((mu + rx - 1)**2 + ry**2 + rz**2)**(9/2) + (90*rz**2*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(7/2) - (105*rz**4*(mu - 1))/((mu + rx)**2 + ry**2 + rz**2)**(9/2)
    dazdrzrzvx = 0
    dazdrzrzvy = 0
    dazdrzrzvz = 0
    dazdrzvxrx = 0
    dazdrzvxry = 0
    dazdrzvxrz = 0
    dazdrzvxvx = 0
    dazdrzvxvy = 0
    dazdrzvxvz = 0
    dazdrzvyrx = 0
    dazdrzvyry = 0
    dazdrzvyrz = 0
    dazdrzvyvx = 0
    dazdrzvyvy = 0
    dazdrzvyvz = 0
    dazdrzvzrx = 0
    dazdrzvzry = 0
    dazdrzvzrz = 0
    dazdrzvzvx = 0
    dazdrzvzvy = 0
    dazdrzvzvz = 0
    dazdvxrxrx = 0
    dazdvxrxry = 0
    dazdvxrxrz = 0
    dazdvxrxvx = 0
    dazdvxrxvy = 0
    dazdvxrxvz = 0
    dazdvxryrx = 0
    dazdvxryry = 0
    dazdvxryrz = 0
    dazdvxryvx = 0
    dazdvxryvy = 0
    dazdvxryvz = 0
    dazdvxrzrx = 0
    dazdvxrzry = 0
    dazdvxrzrz = 0
    dazdvxrzvx = 0
    dazdvxrzvy = 0
    dazdvxrzvz = 0
    dazdvxvxrx = 0
    dazdvxvxry = 0
    dazdvxvxrz = 0
    dazdvxvxvx = 0
    dazdvxvxvy = 0
    dazdvxvxvz = 0
    dazdvxvyrx = 0
    dazdvxvyry = 0
    dazdvxvyrz = 0
    dazdvxvyvx = 0
    dazdvxvyvy = 0
    dazdvxvyvz = 0
    dazdvxvzrx = 0
    dazdvxvzry = 0
    dazdvxvzrz = 0
    dazdvxvzvx = 0
    dazdvxvzvy = 0
    dazdvxvzvz = 0
    dazdvyrxrx = 0
    dazdvyrxry = 0
    dazdvyrxrz = 0
    dazdvyrxvx = 0
    dazdvyrxvy = 0
    dazdvyrxvz = 0
    dazdvyryrx = 0
    dazdvyryry = 0
    dazdvyryrz = 0
    dazdvyryvx = 0
    dazdvyryvy = 0
    dazdvyryvz = 0
    dazdvyrzrx = 0
    dazdvyrzry = 0
    dazdvyrzrz = 0
    dazdvyrzvx = 0
    dazdvyrzvy = 0
    dazdvyrzvz = 0
    dazdvyvxrx = 0
    dazdvyvxry = 0
    dazdvyvxrz = 0
    dazdvyvxvx = 0
    dazdvyvxvy = 0
    dazdvyvxvz = 0
    dazdvyvyrx = 0
    dazdvyvyry = 0
    dazdvyvyrz = 0
    dazdvyvyvx = 0
    dazdvyvyvy = 0
    dazdvyvyvz = 0
    dazdvyvzrx = 0
    dazdvyvzry = 0
    dazdvyvzrz = 0
    dazdvyvzvx = 0
    dazdvyvzvy = 0
    dazdvyvzvz = 0
    dazdvzrxrx = 0
    dazdvzrxry = 0
    dazdvzrxrz = 0
    dazdvzrxvx = 0
    dazdvzrxvy = 0
    dazdvzrxvz = 0
    dazdvzryrx = 0
    dazdvzryry = 0
    dazdvzryrz = 0
    dazdvzryvx = 0
    dazdvzryvy = 0
    dazdvzryvz = 0
    dazdvzrzrx = 0
    dazdvzrzry = 0
    dazdvzrzrz = 0
    dazdvzrzvx = 0
    dazdvzrzvy = 0
    dazdvzrzvz = 0
    dazdvzvxrx = 0
    dazdvzvxry = 0
    dazdvzvxrz = 0
    dazdvzvxvx = 0
    dazdvzvxvy = 0
    dazdvzvxvz = 0
    dazdvzvyrx = 0
    dazdvzvyry = 0
    dazdvzvyrz = 0
    dazdvzvyvx = 0
    dazdvzvyvy = 0
    dazdvzvyvz = 0
    dazdvzvzrx = 0
    dazdvzvzry = 0
    dazdvzvzrz = 0
    dazdvzvzvx = 0
    dazdvzvzvy = 0
    dazdvzvzvz = 0
    """Assign the elements of A"""
    """ax"""
    A[3, 0] = np.array([
        [daxdrxrxrx, daxdrxrxry, daxdrxrxrz, daxdrxrxvx, daxdrxrxvy, daxdrxrxvz],
        [daxdrxryrx, daxdrxryry, daxdrxryrz, daxdrxryvx, daxdrxryvy, daxdrxryvz],
        [daxdrxrzrx, daxdrxrzry, daxdrxrzrz, daxdrxrzvx, daxdrxrzvy, daxdrxrzvz],
        [daxdrxvxrx, daxdrxvxry, daxdrxvxrz, daxdrxvxvx, daxdrxvxvy, daxdrxvxvz],
        [daxdrxvyrx, daxdrxvyry, daxdrxvyrz, daxdrxvyvx, daxdrxvyvy, daxdrxvyvz],
        [daxdrxvzrx, daxdrxvzry, daxdrxvzrz, daxdrxvzvx, daxdrxvzvy, daxdrxvzvz],
    ])
    A[3, 1] = np.array([
        [daxdryrxrx, daxdryrxry, daxdryrxrz, daxdryrxvx, daxdryrxvy, daxdryrxvz],
        [daxdryryrx, daxdryryry, daxdryryrz, daxdryryvx, daxdryryvy, daxdryryvz],
        [daxdryrzrx, daxdryrzry, daxdryrzrz, daxdryrzvx, daxdryrzvy, daxdryrzvz],
        [daxdryvxrx, daxdryvxry, daxdryvxrz, daxdryvxvx, daxdryvxvy, daxdryvxvz],
        [daxdryvyrx, daxdryvyry, daxdryvyrz, daxdryvyvx, daxdryvyvy, daxdryvyvz],
        [daxdryvzrx, daxdryvzry, daxdryvzrz, daxdryvzvx, daxdryvzvy, daxdryvzvz],
    ])
    A[3, 2] = np.array([
        [daxdrzrxrx, daxdrzrxry, daxdrzrxrz, daxdrzrxvx, daxdrzrxvy, daxdrzrxvz],
        [daxdrzryrx, daxdrzryry, daxdrzryrz, daxdrzryvx, daxdrzryvy, daxdrzryvz],
        [daxdrzrzrx, daxdrzrzry, daxdrzrzrz, daxdrzrzvx, daxdrzrzvy, daxdrzrzvz],
        [daxdrzvxrx, daxdrzvxry, daxdrzvxrz, daxdrzvxvx, daxdrzvxvy, daxdrzvxvz],
        [daxdrzvyrx, daxdrzvyry, daxdrzvyrz, daxdrzvyvx, daxdrzvyvy, daxdrzvyvz],
        [daxdrzvzrx, daxdrzvzry, daxdrzvzrz, daxdrzvzvx, daxdrzvzvy, daxdrzvzvz],
    ])
    A[3, 3] = np.array([
        [daxdvxrxrx, daxdvxrxry, daxdvxrxrz, daxdvxrxvx, daxdvxrxvy, daxdvxrxvz],
        [daxdvxryrx, daxdvxryry, daxdvxryrz, daxdvxryvx, daxdvxryvy, daxdvxryvz],
        [daxdvxrzrx, daxdvxrzry, daxdvxrzrz, daxdvxrzvx, daxdvxrzvy, daxdvxrzvz],
        [daxdvxvxrx, daxdvxvxry, daxdvxvxrz, daxdvxvxvx, daxdvxvxvy, daxdvxvxvz],
        [daxdvxvyrx, daxdvxvyry, daxdvxvyrz, daxdvxvyvx, daxdvxvyvy, daxdvxvyvz],
        [daxdvxvzrx, daxdvxvzry, daxdvxvzrz, daxdvxvzvx, daxdvxvzvy, daxdvxvzvz],
    ])
    A[3, 4] = np.array([
        [daxdvyrxrx, daxdvyrxry, daxdvyrxrz, daxdvyrxvx, daxdvyrxvy, daxdvyrxvz],
        [daxdvyryrx, daxdvyryry, daxdvyryrz, daxdvyryvx, daxdvyryvy, daxdvyryvz],
        [daxdvyrzrx, daxdvyrzry, daxdvyrzrz, daxdvyrzvx, daxdvyrzvy, daxdvyrzvz],
        [daxdvyvxrx, daxdvyvxry, daxdvyvxrz, daxdvyvxvx, daxdvyvxvy, daxdvyvxvz],
        [daxdvyvyrx, daxdvyvyry, daxdvyvyrz, daxdvyvyvx, daxdvyvyvy, daxdvyvyvz],
        [daxdvyvzrx, daxdvyvzry, daxdvyvzrz, daxdvyvzvx, daxdvyvzvy, daxdvyvzvz],
    ])
    A[3, 5] = np.array([
        [daxdvzrxrx, daxdvzrxry, daxdvzrxrz, daxdvzrxvx, daxdvzrxvy, daxdvzrxvz],
        [daxdvzryrx, daxdvzryry, daxdvzryrz, daxdvzryvx, daxdvzryvy, daxdvzryvz],
        [daxdvzrzrx, daxdvzrzry, daxdvzrzrz, daxdvzrzvx, daxdvzrzvy, daxdvzrzvz],
        [daxdvzvxrx, daxdvzvxry, daxdvzvxrz, daxdvzvxvx, daxdvzvxvy, daxdvzvxvz],
        [daxdvzvyrx, daxdvzvyry, daxdvzvyrz, daxdvzvyvx, daxdvzvyvy, daxdvzvyvz],
        [daxdvzvzrx, daxdvzvzry, daxdvzvzrz, daxdvzvzvx, daxdvzvzvy, daxdvzvzvz],
    ])
    """ay"""
    A[4, 0] = np.array([
        [daydrxrxrx, daydrxrxry, daydrxrxrz, daydrxrxvx, daydrxrxvy, daydrxrxvz],
        [daydrxryrx, daydrxryry, daydrxryrz, daydrxryvx, daydrxryvy, daydrxryvz],
        [daydrxrzrx, daydrxrzry, daydrxrzrz, daydrxrzvx, daydrxrzvy, daydrxrzvz],
        [daydrxvxrx, daydrxvxry, daydrxvxrz, daydrxvxvx, daydrxvxvy, daydrxvxvz],
        [daydrxvyrx, daydrxvyry, daydrxvyrz, daydrxvyvx, daydrxvyvy, daydrxvyvz],
        [daydrxvzrx, daydrxvzry, daydrxvzrz, daydrxvzvx, daydrxvzvy, daydrxvzvz],
    ])
    A[4, 1] = np.array([
        [daydryrxrx, daydryrxry, daydryrxrz, daydryrxvx, daydryrxvy, daydryrxvz],
        [daydryryrx, daydryryry, daydryryrz, daydryryvx, daydryryvy, daydryryvz],
        [daydryrzrx, daydryrzry, daydryrzrz, daydryrzvx, daydryrzvy, daydryrzvz],
        [daydryvxrx, daydryvxry, daydryvxrz, daydryvxvx, daydryvxvy, daydryvxvz],
        [daydryvyrx, daydryvyry, daydryvyrz, daydryvyvx, daydryvyvy, daydryvyvz],
        [daydryvzrx, daydryvzry, daydryvzrz, daydryvzvx, daydryvzvy, daydryvzvz],
    ])
    A[4, 2] = np.array([
        [daydrzrxrx, daydrzrxry, daydrzrxrz, daydrzrxvx, daydrzrxvy, daydrzrxvz],
        [daydrzryrx, daydrzryry, daydrzryrz, daydrzryvx, daydrzryvy, daydrzryvz],
        [daydrzrzrx, daydrzrzry, daydrzrzrz, daydrzrzvx, daydrzrzvy, daydrzrzvz],
        [daydrzvxrx, daydrzvxry, daydrzvxrz, daydrzvxvx, daydrzvxvy, daydrzvxvz],
        [daydrzvyrx, daydrzvyry, daydrzvyrz, daydrzvyvx, daydrzvyvy, daydrzvyvz],
        [daydrzvzrx, daydrzvzry, daydrzvzrz, daydrzvzvx, daydrzvzvy, daydrzvzvz],
    ])
    A[4, 3] = np.array([
        [daydvxrxrx, daydvxrxry, daydvxrxrz, daydvxrxvx, daydvxrxvy, daydvxrxvz],
        [daydvxryrx, daydvxryry, daydvxryrz, daydvxryvx, daydvxryvy, daydvxryvz],
        [daydvxrzrx, daydvxrzry, daydvxrzrz, daydvxrzvx, daydvxrzvy, daydvxrzvz],
        [daydvxvxrx, daydvxvxry, daydvxvxrz, daydvxvxvx, daydvxvxvy, daydvxvxvz],
        [daydvxvyrx, daydvxvyry, daydvxvyrz, daydvxvyvx, daydvxvyvy, daydvxvyvz],
        [daydvxvzrx, daydvxvzry, daydvxvzrz, daydvxvzvx, daydvxvzvy, daydvxvzvz],
    ])
    A[4, 4] = np.array([
        [daydvyrxrx, daydvyrxry, daydvyrxrz, daydvyrxvx, daydvyrxvy, daydvyrxvz],
        [daydvyryrx, daydvyryry, daydvyryrz, daydvyryvx, daydvyryvy, daydvyryvz],
        [daydvyrzrx, daydvyrzry, daydvyrzrz, daydvyrzvx, daydvyrzvy, daydvyrzvz],
        [daydvyvxrx, daydvyvxry, daydvyvxrz, daydvyvxvx, daydvyvxvy, daydvyvxvz],
        [daydvyvyrx, daydvyvyry, daydvyvyrz, daydvyvyvx, daydvyvyvy, daydvyvyvz],
        [daydvyvzrx, daydvyvzry, daydvyvzrz, daydvyvzvx, daydvyvzvy, daydvyvzvz],
    ])
    A[4, 5] = np.array([
        [daydvzrxrx, daydvzrxry, daydvzrxrz, daydvzrxvx, daydvzrxvy, daydvzrxvz],
        [daydvzryrx, daydvzryry, daydvzryrz, daydvzryvx, daydvzryvy, daydvzryvz],
        [daydvzrzrx, daydvzrzry, daydvzrzrz, daydvzrzvx, daydvzrzvy, daydvzrzvz],
        [daydvzvxrx, daydvzvxry, daydvzvxrz, daydvzvxvx, daydvzvxvy, daydvzvxvz],
        [daydvzvyrx, daydvzvyry, daydvzvyrz, daydvzvyvx, daydvzvyvy, daydvzvyvz],
        [daydvzvzrx, daydvzvzry, daydvzvzrz, daydvzvzvx, daydvzvzvy, daydvzvzvz],
    ])
    """az"""
    A[5, 0] = np.array([
        [dazdrxrxrx, dazdrxrxry, dazdrxrxrz, dazdrxrxvx, dazdrxrxvy, dazdrxrxvz],
        [dazdrxryrx, dazdrxryry, dazdrxryrz, dazdrxryvx, dazdrxryvy, dazdrxryvz],
        [dazdrxrzrx, dazdrxrzry, dazdrxrzrz, dazdrxrzvx, dazdrxrzvy, dazdrxrzvz],
        [dazdrxvxrx, dazdrxvxry, dazdrxvxrz, dazdrxvxvx, dazdrxvxvy, dazdrxvxvz],
        [dazdrxvyrx, dazdrxvyry, dazdrxvyrz, dazdrxvyvx, dazdrxvyvy, dazdrxvyvz],
        [dazdrxvzrx, dazdrxvzry, dazdrxvzrz, dazdrxvzvx, dazdrxvzvy, dazdrxvzvz],
    ])
    A[5, 1] = np.array([
        [dazdryrxrx, dazdryrxry, dazdryrxrz, dazdryrxvx, dazdryrxvy, dazdryrxvz],
        [dazdryryrx, dazdryryry, dazdryryrz, dazdryryvx, dazdryryvy, dazdryryvz],
        [dazdryrzrx, dazdryrzry, dazdryrzrz, dazdryrzvx, dazdryrzvy, dazdryrzvz],
        [dazdryvxrx, dazdryvxry, dazdryvxrz, dazdryvxvx, dazdryvxvy, dazdryvxvz],
        [dazdryvyrx, dazdryvyry, dazdryvyrz, dazdryvyvx, dazdryvyvy, dazdryvyvz],
        [dazdryvzrx, dazdryvzry, dazdryvzrz, dazdryvzvx, dazdryvzvy, dazdryvzvz],
    ])
    A[5, 2] = np.array([
        [dazdrzrxrx, dazdrzrxry, dazdrzrxrz, dazdrzrxvx, dazdrzrxvy, dazdrzrxvz],
        [dazdrzryrx, dazdrzryry, dazdrzryrz, dazdrzryvx, dazdrzryvy, dazdrzryvz],
        [dazdrzrzrx, dazdrzrzry, dazdrzrzrz, dazdrzrzvx, dazdrzrzvy, dazdrzrzvz],
        [dazdrzvxrx, dazdrzvxry, dazdrzvxrz, dazdrzvxvx, dazdrzvxvy, dazdrzvxvz],
        [dazdrzvyrx, dazdrzvyry, dazdrzvyrz, dazdrzvyvx, dazdrzvyvy, dazdrzvyvz],
        [dazdrzvzrx, dazdrzvzry, dazdrzvzrz, dazdrzvzvx, dazdrzvzvy, dazdrzvzvz],
    ])
    A[5, 3] = np.array([
        [dazdvxrxrx, dazdvxrxry, dazdvxrxrz, dazdvxrxvx, dazdvxrxvy, dazdvxrxvz],
        [dazdvxryrx, dazdvxryry, dazdvxryrz, dazdvxryvx, dazdvxryvy, dazdvxryvz],
        [dazdvxrzrx, dazdvxrzry, dazdvxrzrz, dazdvxrzvx, dazdvxrzvy, dazdvxrzvz],
        [dazdvxvxrx, dazdvxvxry, dazdvxvxrz, dazdvxvxvx, dazdvxvxvy, dazdvxvxvz],
        [dazdvxvyrx, dazdvxvyry, dazdvxvyrz, dazdvxvyvx, dazdvxvyvy, dazdvxvyvz],
        [dazdvxvzrx, dazdvxvzry, dazdvxvzrz, dazdvxvzvx, dazdvxvzvy, dazdvxvzvz],
    ])
    A[5, 4] = np.array([
        [dazdvyrxrx, dazdvyrxry, dazdvyrxrz, dazdvyrxvx, dazdvyrxvy, dazdvyrxvz],
        [dazdvyryrx, dazdvyryry, dazdvyryrz, dazdvyryvx, dazdvyryvy, dazdvyryvz],
        [dazdvyrzrx, dazdvyrzry, dazdvyrzrz, dazdvyrzvx, dazdvyrzvy, dazdvyrzvz],
        [dazdvyvxrx, dazdvyvxry, dazdvyvxrz, dazdvyvxvx, dazdvyvxvy, dazdvyvxvz],
        [dazdvyvyrx, dazdvyvyry, dazdvyvyrz, dazdvyvyvx, dazdvyvyvy, dazdvyvyvz],
        [dazdvyvzrx, dazdvyvzry, dazdvyvzrz, dazdvyvzvx, dazdvyvzvy, dazdvyvzvz],
    ])
    A[5, 5] = np.array([
        [dazdvzrxrx, dazdvzrxry, dazdvzrxrz, dazdvzrxvx, dazdvzrxvy, dazdvzrxvz],
        [dazdvzryrx, dazdvzryry, dazdvzryrz, dazdvzryvx, dazdvzryvy, dazdvzryvz],
        [dazdvzrzrx, dazdvzrzry, dazdvzrzrz, dazdvzrzvx, dazdvzrzvy, dazdvzrzvz],
        [dazdvzvxrx, dazdvzvxry, dazdvzvxrz, dazdvzvxvx, dazdvzvxvy, dazdvzvxvz],
        [dazdvzvyrx, dazdvzvyry, dazdvzvyrz, dazdvzvyvx, dazdvzvyvy, dazdvzvyvz],
        [dazdvzvzrx, dazdvzvzry, dazdvzvzrz, dazdvzvzvx, dazdvzvzvy, dazdvzvzvz],
    ])
    """return A"""
    return A

if __name__ == '__main__':
    """Main test"""
    print("RE_STM =", 1)
    print("RE_STT =", 1)