"""Robustness check: harmonic fit and duration asymmetry on the gap-filled record.

Reads  data/processed/hourly.pkl, data/processed/daily.pkl (run daily.py first)
Writes results/robustness_check.csv

Each missing hour is replaced by
    eta(t) = sum_k f_k A_k cos(w_k t + u_k - phi_k) + zeta(t),
where the tidal part is the 33-constituent prediction and zeta is the daily mean
level from GP-2M-SA (observed where a daily value exists, predicted otherwise),
interpolated linearly from noon of each day to the hour. zeta already contains
the mean level Z0. Observed hours are left unchanged. The 33-constituent fit and
the turning-point duration analysis are then repeated on the completed series
and compared with the observed series.

Runtime about two minutes (one GP fit on the daily record).
"""
import numpy as np, pandas as pd
from pathlib import Path
import tide
from models import GPModel, tyears

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT/'data'/'processed', ROOT/'results'
NAMES = list(tide.CONST)

hourly = pd.read_pickle(PROC/'hourly.pkl')
d = pd.read_pickle(PROC/'daily.pkl')
obs_h = hourly.dropna()
coef = tide.fit(obs_h.index, obs_h.values, NAMES)

# ---- 1. daily mean level for every day, from GP-2M-SA ----
t = tyears(d.index); y = d.dmwl.values; ok = ~np.isnan(y)
g = GPModel('GP2M32SA').fit(t[ok], y[ok])
zeta_d = pd.Series(np.where(ok, y, g.predict(t)[0]), index=d.index)
zeta_d.index = zeta_d.index + pd.Timedelta('12h')
zeta_h = (zeta_d.reindex(zeta_d.index.union(hourly.index))
          .interpolate('time').reindex(hourly.index).bfill().ffill())

# ---- 2. completed hourly series ----
miss = hourly.isna()
filled = hourly.copy()
filled[miss] = tide.predict(hourly.index[miss], NAMES, coef) + zeta_h[miss].values
print(f'hours {len(hourly)}, missing {int(miss.sum())} ({100*miss.mean():.1f} %)')

# ---- 3. durations from turning points (same rules as tidal_analysis.py) ----
def durations(s):
    t_, v_ = s.index, s.values
    ext = []
    for i in range(2, len(v_)-2):
        if (t_[i+2]-t_[i-2]).total_seconds() != 4*3600: continue
        w = v_[i-2:i+3]
        if v_[i] == w.max() and v_[i] > v_[i-1] and v_[i] > v_[i+1]: ext.append((t_[i], v_[i], 'H'))
        elif v_[i] == w.min() and v_[i] < v_[i-1] and v_[i] < v_[i+1]: ext.append((t_[i], v_[i], 'L'))
    seq = []
    for e in ext:
        if seq and seq[-1][2] == e[2]:
            if (e[2]=='H' and e[1]>seq[-1][1]) or (e[2]=='L' and e[1]<seq[-1][1]): seq[-1] = e
        else: seq.append(e)
    r, f = [], []
    for a, b in zip(seq[:-1], seq[1:]):
        dt = (b[0]-a[0]).total_seconds()/3600.0
        if 2 <= dt <= 20: (r if a[2]=='L' else f).append(dt)
    return np.array(r), np.array(f)

r0, f0 = durations(obs_h)
r1, f1 = durations(filled)
A0 = tide.amplitudes(NAMES, coef).set_index('constituent')
A1 = tide.amplitudes(NAMES, tide.fit(filled.index, filled.values, NAMES)).set_index('constituent')
dA = 100*(A1.amplitude_m - A0.amplitude_m)

rows = [dict(quantity='n_rise', observed=len(r0), filled=len(r1)),
        dict(quantity='n_fall', observed=len(f0), filled=len(f1)),
        dict(quantity='mean_rise_h', observed=r0.mean(), filled=r1.mean()),
        dict(quantity='mean_fall_h', observed=f0.mean(), filled=f1.mean()),
        dict(quantity='asymmetry_h', observed=f0.mean()-r0.mean(), filled=f1.mean()-r1.mean()),
        dict(quantity='max_abs_amp_change_cm', observed=np.nan, filled=dA.abs().max())]
for n in ('M2', 'K1', 'S2', 'O1', 'M4'):
    rows.append(dict(quantity=f'amp_{n}_cm', observed=100*A0.amplitude_m[n],
                     filled=100*A1.amplitude_m[n]))
T = pd.DataFrame(rows)
T.round(4).to_csv(RES/'robustness_check.csv', index=False)
print(T.round(3).to_string(index=False))
print(f'largest amplitude change in {dA.abs().idxmax()}')
