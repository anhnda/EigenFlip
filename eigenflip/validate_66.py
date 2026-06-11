"""Validate shrinkage H-builders preserve diagonal, and distortion metric."""
import numpy as np
rng = np.random.default_rng(3)

d = 30
A = rng.standard_normal((d, d)); Sigma = A @ A.T / d   # PSD
mu = rng.standard_normal(d)*0.5

def H_cov_shrink(mu, Sigma, lam):
    Sb = (1-lam)*Sigma.copy()
    idx = np.arange(d)
    Sb[idx, idx] = (1-lam)*np.diag(Sigma) + lam*np.diag(Sigma)
    return np.outer(mu,mu) + Sb

def H_2m_shrink(mu, Sigma, lam):
    H = Sigma + np.outer(mu,mu)
    dH = np.diag(H).copy()
    Hb = (1-lam)*H
    idx=np.arange(d); Hb[idx,idx] = (1-lam)*dH + lam*dH
    return Hb

H_full = Sigma + np.outer(mu,mu)
print("=== diagonal preservation (must be exact for all lambda) ===")
for lam in [0.0, 0.01, 0.1, 0.3, 1.0]:
    Hc = H_cov_shrink(mu,Sigma,lam); H2 = H_2m_shrink(mu,Sigma,lam)
    # cov: diag(H_cov) should equal diag(mu mu^T + Sigma) = diag(H_full)
    dc = np.abs(np.diag(Hc)-np.diag(H_full)).max()
    # 2m: diag(H_2m) should equal diag(H_full)
    d2 = np.abs(np.diag(H2)-np.diag(H_full)).max()
    # lam=1: cov off-diag should be only from mu mu^T (Sigma off-diag killed)
    offdiag_cov = Hc - np.diag(np.diag(Hc))
    expected_off = np.outer(mu,mu) - np.diag(np.diag(np.outer(mu,mu))) + (1-lam)*(Sigma-np.diag(np.diag(Sigma)))
    off_err = np.abs(offdiag_cov - expected_off).max()
    print(f"  lam={lam:.2f}: cov diag err={dc:.2e}  2m diag err={d2:.2e}  cov offdiag err={off_err:.2e}")

print("\n=== lambda=1 collapses cov-shrink to mu mu^T + diag(Sigma) ===")
Hc1 = H_cov_shrink(mu,Sigma,1.0)
target = np.outer(mu,mu) + np.diag(np.diag(Sigma))
print(f"  ||H_cov(1) - (mu mu^T + diag Sigma)|| = {np.abs(Hc1-target).max():.2e}")

print("\n=== distortion tr(E H E^T) via ((E@H)*E).sum() matches explicit ===")
C=7; E = rng.standard_normal((C,d))
fast = ((E @ H_full)*E).sum()
slow = sum(E[j] @ H_full @ E[j] for j in range(C))
print(f"  fast={fast:.6f}  slow={slow:.6f}  diff={abs(fast-slow):.2e}")

print("\n=== monotonicity sanity: distortion under H_full as lam sweeps ===")
# encode is fixed here; just confirm scorer is stable/positive
for lam in [0.0,0.1,0.3]:
    Hc = H_cov_shrink(mu,Sigma,lam)
    val = ((E@Hc)*E).sum()
    print(f"  lam={lam}: tr(E H_cov E^T)={val:.4f} (>=0: {val>=0})")
