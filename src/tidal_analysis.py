"""Harmonic constants, duration asymmetry and non-linear distortion.

Reads  data/processed/hourly.pkl
Writes results/constituents_33.csv, results/duration_by_year.csv,
       results/distortion_by_period.csv and three figures.

Runtime about a minute, most of it the bootstrap.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import tide

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT/'results', ROOT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
NAMES = list(tide.CONST)
hourly = pd.read_pickle(ROOT/'data'/'processed'/'hourly.pkl')
h = hourly.dropna()
coef = tide.fit(h.index, h.values, NAMES)
A0 = tide.amplitudes(NAMES, coef)
resid = h.values - (coef[0] + tide.design(h.index, NAMES) @ coef[1:])
print(f'variance explained {1 - resid.var()/h.values.var():.4f}, '
      f'residual sd {100*resid.std():.2f} cm')

# ---- 1. the thirty-three constituents, with moving-block bootstrap errors ----
X = np.column_stack([np.ones(len(h)), tide.design(h.index, NAMES)])
fit0 = X @ coef
rng = np.random.default_rng(7); L, B = 720, 200
nb = int(np.ceil(len(h)/L))
amp_b = np.zeros((B, len(NAMES))); pha_b = np.zeros((B, len(NAMES)))
for b in range(B):
    st = rng.integers(0, len(h)-L, size=nb)
    rb = np.concatenate([resid[s:s+L] for s in st])[:len(h)]
    cb, *_ = np.linalg.lstsq(X, fit0 + rb, rcond=None)
    Ab = tide.amplitudes(NAMES, cb)
    amp_b[b] = Ab.amplitude_m.values; pha_b[b] = Ab.phase_deg_rel2020.values
wrap = ((pha_b - A0.phase_deg_rel2020.values + 180) % 360) - 180
pd.DataFrame({'constituent': NAMES,
              'speed_deg_h': [tide.CONST[n][0] for n in NAMES],
              'amp_cm': 100*A0.amplitude_m.values,
              'amp_se_cm': 100*amp_b.std(0, ddof=1),
              'phase_deg': A0.phase_deg_rel2020.values,
              'phase_se_deg': wrap.std(0, ddof=1)}
             ).sort_values('amp_cm', ascending=False).round(4).to_csv(
    RES/'constituents_33.csv', index=False)

# ---- 2. duration asymmetry ----
t_, v_ = h.index, h.values
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
rt, rv, ft, fv = [], [], [], []
for a, b in zip(seq[:-1], seq[1:]):
    dt = (b[0]-a[0]).total_seconds()/3600.0
    if dt < 2 or dt > 20: continue
    (rt.append(a[0]) or rv.append(dt)) if a[2]=='L' else (ft.append(a[0]) or fv.append(dt))
Rs = pd.Series(rv, index=pd.DatetimeIndex(rt)); Fs = pd.Series(fv, index=pd.DatetimeIndex(ft))
print(f'rise n={len(Rs)} mean {Rs.mean():.2f} h, fall n={len(Fs)} mean {Fs.mean():.2f} h, '
      f'asymmetry {Fs.mean()-Rs.mean():+.2f} h')
rows = []
for y in range(2019, 2027):
    r, f = Rs[Rs.index.year==y], Fs[Fs.index.year==y]
    if len(r) < 50: continue
    rows.append(dict(year=y, n_rise=len(r), rise=r.mean(), se_rise=r.std(ddof=1)/np.sqrt(len(r)),
                     n_fall=len(f), fall=f.mean(), se_fall=f.std(ddof=1)/np.sqrt(len(f)),
                     asym=f.mean()-r.mean(),
                     se_asym=np.sqrt(r.var(ddof=1)/len(r)+f.var(ddof=1)/len(f))))
DUR = pd.DataFrame(rows).set_index('year'); DUR.round(4).to_csv(RES/'duration_by_year.csv')
fu = DUR.loc[2020:2025]
x = fu.index.values.astype(float); w = 1/fu.se_asym.values**2
Xd = np.column_stack([np.ones(len(x)), x-x.mean()])
Cm = np.linalg.inv(Xd.T@np.diag(w)@Xd); bb = Cm@(Xd.T@np.diag(w)@fu.asym.values)
print(f'asymmetry trend {bb[1]:+.3f} +/- {np.sqrt(Cm[1,1]):.3f} h/yr '
      f'(t = {bb[1]/np.sqrt(Cm[1,1]):.2f})')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].errorbar(DUR.index, DUR.fall, DUR.se_fall, marker='s', label='fall')
ax[0].errorbar(DUR.index, DUR.rise, DUR.se_rise, marker='o', label='rise')
ax[0].set_ylabel('hours'); ax[0].set_xlabel('year'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].errorbar(DUR.index, DUR.asym, DUR.se_asym, marker='o', color='k', ls='none')
ax[1].plot(x, bb[0]+bb[1]*(x-x.mean()), color='gray', ls='--')
ax[1].set_ylabel('fall - rise (h)'); ax[1].set_xlabel('year'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/'fig08_duration_asymmetry.png', dpi=120); plt.close()

# ---- 3. M4/M2 and the relative phase ----
def distortion(idx):
    c = tide.fit(idx, hourly.loc[idx].values, NAMES)
    a = tide.amplitudes(NAMES, c).set_index('constituent')
    rel = (2*a.loc['M2','phase_deg_rel2020'] - a.loc['M4','phase_deg_rel2020']) % 360
    return a.loc['M4','amplitude_m']/a.loc['M2','amplitude_m'], rel
ra, re = distortion(h.index)
print(f'whole record M4/M2 {ra:.4f}, relative phase {re:.1f} deg')
rows = [dict(period='whole record', n=len(h), ratio=ra, rel_phase=re)]
for y in range(2020, 2026):
    i = h.index[h.index.year == y]
    a_, r_ = distortion(i); rows.append(dict(period=str(y), n=len(i), ratio=a_, rel_phase=r_))
for y in range(2020, 2026):
    for lab, mos in (('wet', [6,7,8,9,10]), ('dry', [12,1,2,3])):
        i = h.index[(h.index.year==y) & (h.index.month.isin(mos))]
        if len(i) < 2000: continue
        a_, r_ = distortion(i); rows.append(dict(period=f'{y} {lab}', n=len(i), ratio=a_, rel_phase=r_))
for lab, mos in (('wet pooled', [6,7,8,9,10]), ('dry pooled', [12,1,2,3])):
    i = h.index[h.index.month.isin(mos)]
    a_, r_ = distortion(i); rows.append(dict(period=lab, n=len(i), ratio=a_, rel_phase=r_))
DIS = pd.DataFrame(rows).set_index('period'); DIS.round(4).to_csv(RES/'distortion_by_period.csv')
yr = DIS.loc[[str(y) for y in range(2020, 2026)]]
plt.figure(figsize=(4.8, 3.6))
plt.scatter(yr.rel_phase, yr.ratio)
for p, rw in yr.iterrows():
    plt.annotate(p, (rw.rel_phase, rw.ratio), fontsize=8, xytext=(3,3), textcoords='offset points')
plt.scatter([re], [ra], marker='*', s=140, color='k')
plt.xlabel('2*phiM2 - phiM4 (deg)'); plt.ylabel('M4 / M2'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/'fig03_m4m2_distortion.png', dpi=120); plt.close()
print('results and figures written')
