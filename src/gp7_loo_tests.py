"""Paired tests of the leave-one-out residuals against GP-2M-SA.

Reads  results/gp7_full_record.csv and data/processed/loo_*.npy
Writes results/gp7_loo_tests.csv

The squared residuals are paired day by day. They are serially correlated -- the
autocorrelation of the paired difference reaches 0.31 at a lag of two days for
the Matern 3/2 comparison and 0.42 for the Matern 1/2 one, a remnant of the
spring-neap cycle -- so an ordinary paired t test overstates significance.
Newey-West with a lag of eight days and a moving-block bootstrap both widen the
standard error.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
full = pd.read_csv(ROOT / 'results' / 'gp7_full_record.csv')
LOO = {r.key: np.load(PROC / f'loo_{r.key}.npy') for r in full.itertuples()}
LAB = dict(zip(full.key, full.kernel))

def newey_west(d, lag=8):
    n = len(d); dm = d - d.mean(); s = dm @ dm / n
    for L in range(1, lag+1):
        s += 2*(1 - L/(lag+1))*(dm[:-L] @ dm[L:] / n)
    return np.sqrt(s/n)

def block_boot(d, L=30, B=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n/L)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n-L, size=nb)
        out[b] = np.concatenate([d[s:s+L] for s in st])[:n].mean()
    return out.std(ddof=1)

ref = LOO['GP2M32SA']; rows = []
for k in LOO:
    if k == 'GP2M32SA': continue
    d = LOO[k]**2 - ref**2
    t_n = d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    t_h = d.mean()/newey_west(d)
    t_b = d.mean()/block_boot(d)
    dm = d - d.mean()
    rows.append(dict(kernel=LAB[k], mean_diff_cm2=1e4*d.mean(),
                     t_naive=t_n, p_naive=2*(1-stats.norm.cdf(abs(t_n))),
                     t_hac=t_h,   p_hac=2*(1-stats.norm.cdf(abs(t_h))),
                     t_boot=t_b,  p_boot=2*(1-stats.norm.cdf(abs(t_b))),
                     acf_lag1=np.corrcoef(dm[:-1], dm[1:])[0,1],
                     acf_lag2=np.corrcoef(dm[:-2], dm[2:])[0,1], nw_lag=8))
out = pd.DataFrame(rows)
out.round(6).to_csv(ROOT / 'results' / 'gp7_loo_tests.csv', index=False)
print(out.round(4).to_string(index=False))
