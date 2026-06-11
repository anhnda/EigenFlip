"""
Validate the load-bearing claim: EigenFlip Solve (Woodbury, no d x d) produces
the SAME integer codes as dense GPTQ sequential conditioning on the materialized
H~ = D + V V^T. Pure numpy port of both algorithms.
"""
import numpy as np

rng = np.random.default_rng(0)

def make_problem(d=40, C=7, kp1=4, bits=4, gsize=8):
    W = rng.standard_normal((C, d)).astype(np.float64)
    # group-wise asym scales/zp (replicated within group)
    n_groups = (d + gsize - 1)//gsize
    pin = n_groups*gsize
    Wp = np.zeros((C, pin)); Wp[:, :d] = W
    Wg = Wp.reshape(C, n_groups, gsize)
    wmin = Wg.min(2, keepdims=True); wmax = Wg.max(2, keepdims=True)
    max_int = 2**bits-1
    sc_g = np.clip((wmax-wmin)/max_int, 1e-8, None)
    zp_g = np.clip(np.round(-wmin/sc_g), 0, max_int)
    scale = np.repeat(sc_g, gsize, axis=2).reshape(C, pin)
    zp = np.repeat(zp_g, gsize, axis=2).reshape(C, pin)
    # trust region: random PD D + V V^T on padded width
    D = np.abs(rng.standard_normal(pin)) + 0.5
    V = rng.standard_normal((pin, kp1)) * 0.3
    return Wp, scale, zp, D, V, max_int, d, pin

def order_leverage(D, V):
    lev = (1.0/D) * (V*V).sum(1)
    return list(np.argsort(-lev))

def dense_gptq(Wf, scale, zp, D, V, max_int, order):
    """
    True GPTQ semantics: at each step condition on the CURRENT remaining set R.
    The compensation for r in R uses [H_RR^{-1}]_{r,i}/[H_RR^{-1}]_{ii}, where
    H_RR is the principal submatrix on the not-yet-quantized coordinates.
    Equivalent (and how GPTQ implements it) to a running Schur complement; here
    we recompute the inverse on R each step -- O(d^4) but it's a reference.
    """
    H = np.diag(D) + V @ V.T
    W = Wf.copy(); C, pin = W.shape
    codes = np.zeros((C, pin), dtype=np.int64)
    remaining = list(order)
    for step, i in enumerate(order):
        R = remaining[step:]                 # current not-yet-quantized set
        si, zpi = scale[:, i], zp[:, i]
        q = np.clip(np.round(W[:, i]/si + zpi), 0, max_int)
        wdq = (q-zpi)*si
        e = wdq - W[:, i]
        codes[:, i] = q.astype(np.int64)
        Rrest = R[1:]                         # exclude i itself
        if not Rrest:
            continue
        HRR = H[np.ix_(R, R)]
        HRRinv = np.linalg.inv(HRR)
        # i is index 0 within R
        factor = HRRinv[1:, 0] / HRRinv[0, 0]   # [len(Rrest)]
        W[:, Rrest] -= e[:, None]*factor[None, :]
    return codes

def eigenflip_solve(Wf, scale, zp, D, V, max_int, order):
    C, pin = Wf.shape; kp1 = V.shape[1]
    Dinv = 1.0/D
    M = np.eye(kp1) + (V.T*Dinv) @ V
    Minv = np.linalg.inv(M)
    G = np.zeros((C, kp1))
    codes = np.zeros((C, pin), dtype=np.int64)
    for i in order:
        Vi = V[i]; Di_inv = Dinv[i]
        si, zpi = scale[:, i], zp[:, i]
        comp = Di_inv * (G @ Vi)
        wt = Wf[:, i] + comp
        q = np.clip(np.round(wt/si + zpi), 0, max_int)
        wdq = (q-zpi)*si
        e = wdq - wt
        codes[:, i] = q.astype(np.int64)
        MinvVi = Minv @ Vi
        quad = Vi @ MinvVi
        Hinv_ii = Di_inv*(1.0 - Di_inv*quad)
        if Hinv_ii <= 1e-30: Hinv_ii = Di_inv
        dir_vec = MinvVi*(Di_inv/Hinv_ii)
        G += np.outer(e, dir_vec)
        # downdate
        M -= Di_inv*np.outer(Vi, Vi)
        Ainv_u = Minv @ Vi
        denom = 1.0 - Di_inv*(Vi @ Ainv_u)
        if abs(denom) > 1e-12:
            Minv = Minv + (Di_inv/denom)*np.outer(Ainv_u, Ainv_u)
    return codes

for trial in range(5):
    Wp, scale, zp, D, V, max_int, d, pin = make_problem()
    order = order_leverage(D, V)
    c1 = dense_gptq(Wp, scale, zp, D, V, max_int, order)
    c2 = eigenflip_solve(Wp, scale, zp, D, V, max_int, order)
    agree = (c1 == c2).mean()
    maxdiff = np.abs(c1-c2).max()
    print(f"trial {trial}: code agreement = {agree*100:.2f}%  max|diff| = {maxdiff}")
