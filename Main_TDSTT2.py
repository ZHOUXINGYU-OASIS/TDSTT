import numpy as np
import scipy
from scipy.integrate import solve_ivp
from Module_STT import STM_pred, STT2_pred
from Module_CRTBP import CRTBP_dynamics, cal_1st_tensor, cal_2nd_tensor
import time

def derivative_eigenvector(K, dK, x, lam, dlam):
    """Calculate the derivatives of a given eigenvector"""
    """
    Input: K    -> Matrix
           dK   -> Derivative of the matrix K
           x    -> Eigenvector of the matrix K
           lam  -> Eigenvalue of the eigenvector x
           dlam -> Derivative of the eigenvalue lam
    Output: dx  -> Derivative of the eigenvector x
    """
    dim = len(x)  # dimension of the eigenvector x
    x = x.reshape(dim, 1)
    M = np.eye(dim)
    f = (dlam * M - dK) @ x
    G = K - lam * M
    k = np.where(abs(x) == max(abs(x)))[0]
    G_ = G
    f_ = f
    G_[k] = np.zeros([dim])
    G_[:, k] = np.zeros([dim, 1])
    G_[k, k] = 1
    f_[k] = 0
    v = np.linalg.inv(G_) @ f_
    c = -v.T @ M @ x
    dx = v + c * x
    return dx

def CRTBP_STT2_dynamics(t, y, mu):
    """the dyanmics of the CRTBP model (with STM and STT)"""
    DIM = 6
    x = y[:DIM]
    STM = y[DIM:(DIM ** 2 + DIM)].reshape(DIM, DIM)
    STT = y[(DIM ** 2 + DIM):].reshape(DIM, DIM, DIM)
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
    dy = np.concatenate((dxdt, dSTM, dSTT))
    return dy

def CRTBP_TDSTT2_dynamics(t, y, mu, num):
    """the dyanmics of the CRTBP model (with STM and TDSTT)"""
    DIM = 6
    x = y[:DIM]  # nominal orbit
    STM = y[DIM:(DIM ** 2 + DIM)].reshape(DIM, DIM)  # full STM
    eigenValue = np.exp(y[(DIM ** 2 + DIM):(DIM ** 2 + DIM + num)])  # eigenvalue
    eigenVector = y[(DIM ** 2 + DIM + num):(DIM ** 2 + DIM + num + num * DIM)].reshape(num, DIM)  # eigenVector
    DSTT = y[(DIM ** 2 + DIM + num + num * DIM):].reshape(DIM, num, num)  # directional STT
    """x"""
    dxdt = CRTBP_dynamics(t, x, mu)
    """STM"""
    N1 = cal_1st_tensor(x, mu)
    dSTM = np.matmul(N1, STM)
    """Derivatives of eigenvalues"""
    CGT = STM.T @ STM
    dCGT = dSTM.T @ STM + STM.T @ dSTM
    dvalue = np.zeros([num])
    for i in range(num):
        ni = eigenVector[i]  # i-th eigenvector
        dvalue[i] = np.dot(dCGT @ ni, ni)
    """Derivatives of eigenvectors"""
    dvector = np.zeros([num, DIM])
    Coeff = np.zeros([num, num])  # coefficients
    for i in range(num):
        ni = eigenVector[i]  # i-th eigenvector
        lami = eigenValue[i]  # i-th eigenvalue
        for j in range(num):
            nj = eigenVector[j]  # j-th eigenvector
            lamj = eigenValue[j]  # j-th eigenvalue
            if i == j:
                continue
            else:
                Coeff[i, j] = 1 / (lami - lamj) * np.dot(dCGT @ ni, nj)
        dvector[i] = derivative_eigenvector(CGT, dCGT, ni, lami, dvalue[i]).T[0]
        dvalue[i] = 1 / eigenValue[i] * dvalue[i]  # update eigenvalue derivatives
    R = eigenVector
    B = Coeff
    """DSTM"""
    DSTM = np.zeros([DIM, num])
    for i in range(DIM):
        for k1 in range(num):
            for l1 in range(DIM):
                DSTM[i, k1] += STM[i, l1] * R[k1, l1]
    """DSTT"""
    N2 = cal_2nd_tensor(x, mu)
    dDSTT = np.zeros([DIM, num, num])
    for i in range(DIM):
        for p1 in range(num):
            for p2 in range(num):
                for k1 in range(DIM):
                    dDSTT[i, p1, p2] += N1[i, k1] * DSTT[k1, p1, p2]
                    for k2 in range(DIM):
                        dDSTT[i, p1, p2] += N2[i, k1, k2] * DSTM[k1, p1] * DSTM[k2, p2]
                for gamma1 in range(num):  # Method #3
                    dDSTT[i, p1, p2] += DSTT[i, gamma1, p2] * B[p1, gamma1]
                for gamma2 in range(num):
                    dDSTT[i, p1, p2] += DSTT[i, p1, gamma2] * B[p2, gamma2]
    dSTM = dSTM.reshape(DIM ** 2)
    dDSTT = dDSTT.reshape(DIM * num * num)
    dvector = dvector.reshape(num * DIM)
    dy = np.concatenate((dxdt, dSTM, dvalue, dvector, dDSTT))
    return dy

def generate_TDSTT2(x0, t0, tf, dt, num, len_eval):
    """Generate second-order DSTT terms"""
    print("===== TDSTT =====")
    DIM = 6
    time_cost = np.zeros([2])
    """Propagate the DSTT"""
    t_eval = [t0, (t0 + dt)]
    start = time.time()
    y0_STT = np.concatenate((x0,
                             np.eye(DIM).reshape(DIM ** 2),
                             np.zeros([DIM, DIM, DIM]).reshape(DIM ** 3)))
    """First propagate one step"""
    sol = solve_ivp(CRTBP_STT2_dynamics, [t0, (t0 + dt)], y0_STT, args=(mu,), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)

    x0 = sol.y.T[-1, :DIM]  # nominal orbit
    STM = sol.y.T[-1, DIM:(DIM ** 2 + DIM)].reshape(DIM, DIM)  # full STM
    STT = sol.y.T[-1, (DIM ** 2 + DIM):(DIM ** 3 + DIM ** 2 + DIM)].reshape(DIM, DIM, DIM) # full second-order STT
    """Find the sensitive direction"""
    CGT = STM.T @ STM
    eigenvalue, featurevector = np.linalg.eig(CGT)
    id = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[id]
    featurevector = featurevector.T[id]
    eigenvalue = eigenvalue[:num]
    R = featurevector[:num]
    """Generate DSTTs"""
    STM0 = STM
    DSTT0 = np.zeros([DIM, num, num])
    for i in range(DIM):
        for k1 in range(num):
            for k2 in range(num):
                for p1 in range(DIM):
                    for p2 in range(DIM):
                        DSTT0[i, k1, k2] += STT[i, p1, p2] * R[k1, p1] * R[k2, p2]
    time_cost[0] = time.time() - start
    """Propagate the remaining steps"""
    start = time.time()
    t_eval = np.linspace(t0 + dt, tf, len_eval)
    y0_TDSTT = np.concatenate((x0,
                               STM0.reshape(DIM ** 2),
                               np.log(eigenvalue),
                               R.reshape(DIM * num),  # eigenvector
                               DSTT0.reshape(DIM * num * num)))
    sol = solve_ivp(CRTBP_TDSTT2_dynamics, [(t0 + dt), tf], y0_TDSTT, args=(mu, num), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    """Collect results"""
    STM = np.zeros([len_eval, DIM, DIM])
    DSTT = np.zeros([len_eval, DIM, num, num])
    eigenValue = np.zeros([len_eval, num])
    eigenVector = np.zeros([len_eval, num, DIM])
    for k in range(len_eval):
        STM[k] = sol.y.T[k, DIM:(DIM ** 2 + DIM)].reshape(DIM, DIM)
        DSTT[k] = sol.y.T[k, (DIM ** 2 + DIM + num + num * DIM):].reshape(DIM, num, num)
        eigenValue[k] = np.exp(sol.y.T[k, (DIM ** 2 + DIM):(DIM ** 2 + DIM + num)])  # eigenvalue
        eigenVector[k] = sol.y.T[k, (DIM ** 2 + DIM + num):(DIM ** 2 + DIM + num + num * DIM)].reshape(num, DIM)  # eigenVector
    R = eigenVector
    time_cost[1] = time.time() - start
    return time_cost, STM, DSTT, R, eigenValue

if __name__ == '__main__':
    """Main test"""
    """Load data"""
    data = scipy.io.loadmat("JupiterCase2.mat")
    mu = data["miu"][0, 0]
    x0 = data["x0"][0]
    """Propagate orbit"""
    t0 = data["t0"][0, 0]
    tf = data["tf"][0, 0]
    len_eval = 100000
    dt = (tf - t0) / len_eval
    t_eval = np.linspace(t0, tf, len_eval + 1)
    RelTol = 10 ** -8
    AbsTol = 10 ** -8
    print("===== Nominal orbit =====")
    check = solve_ivp(CRTBP_dynamics, [t0, tf], x0, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xf = check.y.T[-1, :]
    """Propagate deviation orbit"""
    errR = 1.3e-7  # approximated 100 km
    errV = 7.6e-7  # approximated 10 mm/s
    dx0 = np.array([errR, errR, errR, errV, errV, errV])
    print("===== Deviation orbit =====")
    check = solve_ivp(CRTBP_dynamics, [t0, tf], x0 + dx0, args=(mu,), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xf_ = check.y.T[-1, :]
    """Propagate the STT"""
    num = 2
    time_cost, STM, DSTT, R, eigenValue = generate_TDSTT2(x0, t0, tf, dt, num, len_eval)
    """Accuracy validation"""
    dxf = xf_ - xf
    dxf_R = R[-1] @ dx0
    dxf_STM = STM_pred(STM[-1], dx0)
    dxf_DSTT2 = STM_pred(STM[-1], dx0) + STT2_pred(DSTT[-1], dxf_R, dxf_R)
    RE_STM = abs(dxf - dxf_STM) / abs(dxf) * 100
    RE_DSTT = abs(dxf - dxf_DSTT2) / abs(dxf) * 100
    print("RE_STM =", RE_STM)
    print("RE_DSTT =", RE_DSTT)
    """Save data"""
    fileName = "results_TDSTT2_%d.npz" % num
    np.savez(fileName, STM, DSTT, R, time_cost, x0, t0, tf, dt, num)