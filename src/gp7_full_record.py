"""Fit the seven covariance structures to the whole daily record.

Writes results/gp7_full_record.csv

Leave-one-out residuals come from a single matrix inversion rather than n refits.
With a vague prior on beta,  r_i = [Kt^-1 y]_i / [Kt^-1]_ii  with
Kt = K_y + H Sigma_beta H'.  Using the profiled Ainv in place of the vague prior
gives the wrong answer, which is easy to do by accident.

Runtime about seven minutes: each fit is O(n^3) on n = 2412 days.
"""
import numpy as np, pandas as pd, time
from pathlib import Path
from models import GPModel, basis, tyears

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_pickle(ROOT / 'data' / 'processed' / 'daily.pkl')
t = tyears(d.index); y = d.dmwl.values
obs = ~np.isnan(y); tt, yy = t[obs], y[obs]
R = np.abs(tt[:, None] - tt[None, :])

VAR = [('GPM12', 'Matern 1/2'), ('GP', 'Matern 3/2'), ('GPM52', 'Matern 5/2'),
       ('GPRQ', 'rational quadratic'), ('GP2M32', 'two-scale M3/2'),
       ('GP2M32SA', 'GP-2M-SA'), ('GPQP', 'M3/2 + quasi-periodic')]
rows = []
for k, lab in VAR:
    t0 = time.time()
    g = GPModel(k).fit(tt, yy)
    K, _ = g._prep(g.lp, R)
    Kt = K + g.H @ (1e4*np.eye(g.H.shape[1])) @ g.H.T
    Kti = np.linalg.inv(Kt)
    r = (Kti @ yy)/np.diag(Kti)
    np.save(ROOT / 'data' / 'processed' / f'loo_{k}.npy', r)
    rows.append(dict(key=k, kernel=lab, n_hyper=len(g.lp), n_basis=g.H.shape[1],
                     LOO_cm=100*np.sqrt(np.mean(r**2)), AIC=2*g.nll + 2*g.k, nll=g.nll,
                     rate_mm_yr=1000*g.rate()[0], rate_se_mm_yr=1000*g.rate()[1],
                     **{f'p_{a}': b for a, b in g.params.items()}, sec=time.time()-t0))
    print(f'{lab:24s} LOO {rows[-1]["LOO_cm"]:6.3f} cm   AIC {rows[-1]["AIC"]:9.1f}   '
          f'rate {rows[-1]["rate_mm_yr"]:+.2f} +- {rows[-1]["rate_se_mm_yr"]:.2f} mm/yr '
          f'  {rows[-1]["sec"]:.0f}s', flush=True)
pd.DataFrame(rows).round(6).to_csv(ROOT / 'results' / 'gp7_full_record.csv', index=False)
print('written to results/gp7_full_record.csv')
