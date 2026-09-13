"""Seasonal modulation of the tidal constituents.

Reads  data/processed/hourly.pkl
Writes results/seasonal_modulation.csv

Reproduces the numbers of Sections 2.5 and 3.3: the residual standard
deviation with and without annual modulation, the mean amplitude, the
seasonal swing and the month of peak amplitude for each modulated
constituent, and the moving-block bootstrap intervals for the swing.

Also demonstrates the degeneracy that forces K1, S2 and P1 out of the
modulated set.

Runtime about two minutes, almost all of it the bootstrap.
"""
import numpy as np, pandas as pd
from pathlib import Path
import tide

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results'; RES.mkdir(exist_ok=True)
NAMES = list(tide.CONST)
MOD = list(tide.MOD_NAMES)
N_BOOT, BLOCK_DAYS, SEED = 200, 30, 0
REPORT = ['M2','N2','O1','K2','M4','MS4','MN4','MK3','MO3','M6']

h = pd.read_pickle(ROOT/'data'/'processed'/'hourly.pkl').dropna()
t, y = h.index, h.values

def design(mod_names):
    return np.column_stack([np.ones(len(t)), tide.design_mod(t, NAMES, mod_names=mod_names)])

def envelope(coef, mod_names, name, n=366*24):
    """Amplitude of one constituent through a year, in cm."""
    j = NAMES.index(name); a, b = coef[1+2*j], coef[2+2*j]
    k = list(mod_names).index(name); n0 = 1 + 2*len(NAMES)
    p1, p2, p3, p4 = coef[n0+4*k : n0+4*k+4]
    ty = np.linspace(0, 1, n, endpoint=False)
    sa, ca = np.sin(2*np.pi*ty), np.cos(2*np.pi*ty)
    return ty, 100*np.hypot(a + p1*sa + p2*ca, b + p3*sa + p4*ca)

# ---- 1. plain against modulated fit ----
X0 = np.column_stack([np.ones(len(t)), tide.design(t, NAMES)])
c0, *_ = np.linalg.lstsq(X0, y, rcond=None)
Xm = design(MOD)
cm, *_ = np.linalg.lstsq(Xm, y, rcond=None)
print(f'residual sd  without modulation {100*(y-X0@c0).std(ddof=1):.2f} cm'
      f'   with modulation {100*(y-Xm@cm).std(ddof=1):.2f} cm')
print(f'condition number of the modulated design {np.linalg.cond(Xm):.1f}')

# ---- 2. the degeneracy, for the record ----
BAD = ('K1','O1','M2','S2','P1','K2','N2','Q1','M4','MS4','MK3','MO3')
Xb = design(BAD)
cb, *_ = np.linalg.lstsq(Xb, y, rcond=None)
print(f'\nmodulating K1, S2 and P1 as well gives condition number {np.linalg.cond(Xb):.3g}')
for n in ('S2','P1'):
    print(f'   {n} envelope peaks at {envelope(cb, BAD, n)[1].max()/100:.1f} m, physically meaningless')

# ---- 3. swing, peak month and bootstrap interval ----
def swings(coef):
    out = {}
    for n in REPORT:
        ty, A = envelope(coef, MOD, n)
        out[n] = (A.mean(), A.max()-A.min(), ty[A.argmax()])
    return out

obs = swings(cm)
rng = np.random.default_rng(SEED)
resid = y - Xm@cm
day = (t - t[0]).days.values
blocks = [np.flatnonzero(day//BLOCK_DAYS == b) for b in range(day.max()//BLOCK_DAYS + 1)]
blocks = [b for b in blocks if len(b)]
boot = {n: [] for n in REPORT}
for _ in range(N_BOOT):
    idx = np.empty(0, dtype=int)
    while len(idx) < len(y):
        idx = np.concatenate([idx, blocks[rng.integers(0, len(blocks))]])
    yb = Xm@cm + resid[idx[:len(y)]]
    cbst, *_ = np.linalg.lstsq(Xm, yb, rcond=None)
    for n, v in swings(cbst).items():
        boot[n].append(v[1])

rows = []
for n in REPORT:
    mean, swing, typeak = obs[n]
    lo, hi = np.percentile(boot[n], [2.5, 97.5])
    peak = (pd.Timestamp('2020-01-01') + pd.to_timedelta(typeak*365.25, unit='D')).strftime('%b')
    rows.append(dict(constituent=n, mean_cm=round(mean,2), swing_cm=round(swing,2),
                     lo_cm=round(lo,2), hi_cm=round(hi,2),
                     rel_swing_pct=round(100*swing/mean,1), peak_month=peak))
out = pd.DataFrame(rows)
print('\n', out.to_string(index=False))
out.to_csv(RES/'seasonal_modulation.csv', index=False)
print(f'\nwritten to {RES/"seasonal_modulation.csv"}')
