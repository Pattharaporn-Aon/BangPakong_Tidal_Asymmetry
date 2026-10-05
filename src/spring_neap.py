"""Overtides over the spring-neap cycle, by season, and the ratio a/h.

Reads  data/processed/hourly.pkl and data/processed/clean_10min.pkl
Writes results/spring_neap_overtides.csv and results/a_over_h.csv

A model with a constant, a linear term and one harmonic in each of the diurnal,
semidiurnal, terdiurnal, quarter-diurnal and sixth-diurnal bands is fitted by
least squares to every 49-hour window centred on a calendar day with at least
45 valid hours. D4/D2 and D6/D2 are the ratios of the band amplitudes. Within
each 15-day window the days in the upper third of the semidiurnal amplitude are
springs and those in the lower third are neaps. Standard errors come from 15-day
block means, which allows for the serial correlation of consecutive days.

Runtime under a minute.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results'; RES.mkdir(exist_ok=True)
h = pd.read_pickle(ROOT/'data'/'processed'/'hourly.pkl')

BANDS = [('D1', 15.0), ('D2', 29.5), ('D3', 44.0), ('D4', 59.0), ('D6', 88.5)]
WET, DRY = [6, 7, 8, 9, 10], [12, 1, 2, 3]

# ---- 1. band amplitudes from 49-hour fits, one per day ----
rows = []
for d in pd.date_range(h.index[0].ceil('D'), h.index[-1].floor('D'), freq='D'):
    s = h[d - pd.Timedelta('24h'): d + pd.Timedelta('24h')].dropna()
    if len(s) < 45:
        continue
    t = (s.index - d).total_seconds().values/3600.0
    X = [np.ones_like(t), t]
    for _, w in BANDS:
        X += [np.cos(np.deg2rad(w)*t), np.sin(np.deg2rad(w)*t)]
    c, *_ = np.linalg.lstsq(np.column_stack(X), s.values, rcond=None)
    a = {n: np.hypot(c[2+2*i], c[3+2*i]) for i, (n, _) in enumerate(BANDS)}
    a['date'] = d
    rows.append(a)
D = pd.DataFrame(rows).set_index('date')
D['r4'], D['r6'] = D.D4/D.D2, D.D6/D.D2

# ---- 2. spring and neap days, and the season ----
roll = D.D2.rolling('15D', center=True, min_periods=10)
q1, q2 = roll.quantile(1/3), roll.quantile(2/3)
D['phase'] = np.where(D.D2 >= q2, 'spring', np.where(D.D2 <= q1, 'neap', 'mid'))
m = D.index.month
D['season'] = np.where(np.isin(m, WET), 'wet', np.where(np.isin(m, DRY), 'dry', 'other'))

def block_mean(v):
    b = v.groupby((v.index - v.index[0]).days // 15).mean()
    return v.mean(), b.std(ddof=1)/np.sqrt(len(b))

out = []
for se in ('wet', 'dry'):
    for ph in ('spring', 'neap'):
        sel = D[(D.season == se) & (D.phase == ph)]
        r4, e4 = block_mean(sel.r4); r6, e6 = block_mean(sel.r6)
        out.append(dict(season=se, phase=ph, days=len(sel), D2_m=sel.D2.mean(),
                        D4_D2=r4, D4_D2_se=e4, D6_D2=r6, D6_D2_se=e6))
T = pd.DataFrame(out)
T.round(4).to_csv(RES/'spring_neap_overtides.csv', index=False)
print(T.round(4).to_string(index=False))

# ---- 3. a/h by season ----
# a is half the mean daily range from the ten-minute record. h is the dry-season
# depth below mean level, and the wet-season mean level is 14.8 cm lower.
s10 = pd.read_pickle(ROOT/'data'/'processed'/'clean_10min.pkl')
g = s10.groupby(s10.index.floor('D'))
rng = (g.max() - g.min())[g.count() >= 130]
a_wet = rng[rng.index.month.isin(WET)].mean()/2
a_dry = rng[rng.index.month.isin(DRY)].mean()/2
dz = 0.148
rows = []
for depth in (5.0, 10.0, 12.5, 15.0):
    rows.append(dict(depth_m=depth, a_dry_m=a_dry, a_wet_m=a_wet,
                     a_over_h_dry=a_dry/depth, a_over_h_wet=a_wet/(depth - dz),
                     change_pct=100*((a_wet/(depth - dz))/(a_dry/depth) - 1)))
A = pd.DataFrame(rows)
A.round(4).to_csv(RES/'a_over_h.csv', index=False)
print('\n' + A.round(3).to_string(index=False))
