"""Numpy check: flip encoder prefix rule reduces ||z||^2 and respects budget."""
import numpy as np
rng = np.random.default_rng(1)

def flip_encode(Wint, scale, zp, pre, V, max_int, Wf, budget_frac, d):
    C, pin = Wint.shape
    Wint = Wint.astype(float).copy()
    w_dq = (Wint - zp)*scale
    e = w_dq - Wf
    z = e @ V                                  # [C, kp1]
    resid = pre - Wint
    flip_dir = np.sign(resid); flip_dir[flip_dir==0]=1
    proposed = Wint + flip_dir
    in_range = (proposed>=0)&(proposed<=max_int)
    de = flip_dir*scale
    regret = np.where(in_range, np.abs(resid), -1.0)
    order = np.argsort(-regret, axis=1)
    de_s = np.take_along_axis(de, order, 1)
    V_s = V[order]                             # [C, pin, kp1]
    valid_s = np.take_along_axis(in_range.astype(int), order, 1)
    dz = de_s[:,:,None]*V_s*valid_s[:,:,None]
    z_path = z[:,None,:] + np.cumsum(dz, axis=1)
    norm_path = (z_path**2).sum(2)
    z0 = (z**2).sum(1, keepdims=True)
    alln = np.concatenate([z0, norm_path], 1)
    best_m = np.argmin(alln, 1)
    cap = max(1, int(budget_frac*d))
    best_m = np.minimum(best_m, cap)
    idx = np.arange(pin)[None,:]
    accept = (idx<best_m[:,None]) & valid_s.astype(bool)
    fd_s = np.take_along_axis(flip_dir, order, 1)
    applied = np.where(accept, fd_s, 0)
    np.add.at  # noop
    Wint2 = Wint.copy()
    for c in range(C):
        Wint2[c, order[c]] += applied[c]
    Wint2 = np.clip(Wint2, 0, max_int)
    # new ||z||
    e2 = (Wint2-zp)*scale - Wf
    z2 = e2 @ V
    return (z**2).sum(1), (z2**2).sum(1), accept.sum(1), cap

d=32; C=5; kp1=3; bits=4; g=8
W = rng.standard_normal((C,d))
n_groups=d//g
Wg=W.reshape(C,n_groups,g); wmin=Wg.min(2,keepdims=True); wmax=Wg.max(2,keepdims=True)
mi=2**bits-1; scg=np.clip((wmax-wmin)/mi,1e-8,None); zpg=np.clip(np.round(-wmin/scg),0,mi)
scale=np.repeat(scg,g,2).reshape(C,d); zp=np.repeat(zpg,g,2).reshape(C,d)
pre=W/scale+zp; Wint=np.clip(np.round(pre),0,mi)
V=rng.standard_normal((d,kp1))*0.3
z0,z1,nflip,cap=flip_encode(Wint,scale,zp,pre,V,mi,W,1.0,d)
print("budget=1.0 (full):")
print("  ||z||^2 before:", np.round(z0,4))
print("  ||z||^2 after :", np.round(z1,4))
print("  reduced all rows:", bool((z1<=z0+1e-9).all()), " flips/row:", nflip, "cap:",cap)
z0,z1,nflip,cap=flip_encode(Wint,scale,zp,pre,V,mi,W,0.1,d)
print("budget=0.1:")
print("  reduced all rows:", bool((z1<=z0+1e-9).all()), " flips/row:", nflip, "cap:",cap, " (<=cap:",bool((nflip<=cap).all()),")")
