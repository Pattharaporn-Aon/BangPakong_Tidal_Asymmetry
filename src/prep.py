"""Quality control of the ten-minute record and construction of the hourly series.

Reads  data/raw/Clean_BangPaKong_2.xlsx
Writes data/processed/clean_10min.pkl, data/processed/hourly.pkl, results/qc_log.csv

The filters are applied in the order below; each one is reported in the log so
that the effect of every rule can be seen.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / 'data' / 'raw' / 'Clean_BangPaKong_2.xlsx'
OUT  = ROOT / 'data' / 'processed'; OUT.mkdir(parents=True, exist_ok=True)
RES  = ROOT / 'results'; RES.mkdir(exist_ok=True)

raw = pd.read_excel(RAW)
raw = raw.rename(columns={raw.columns[0]: 't', raw.columns[1]: 'h'})
raw['t'] = pd.to_datetime(raw['t'], errors='coerce')
raw = raw.dropna(subset=['t'])
log = {'raw_records': len(raw)}

# 1 duplicate timestamps -> median; drop a timestamp whose duplicates disagree by > 0.10 m
g = raw.groupby('t').h.agg(['median', 'min', 'max', 'count'])
log['duplicate_timestamps'] = int((g['count'] > 1).sum())
bad_dup = (g['max'] - g['min']) > 0.10
log['conflicting_duplicates_removed'] = int(bad_dup.sum())
s = g.loc[~bad_dup, 'median']

# 2 zeros: the logger writes 0 when it has no reading
z = s <= 0; log['zero_values_removed'] = int(z.sum()); s = s[~z]

# 3 saturated high readings
hi = s >= 5.60; log['stuck_high_values_removed(>=5.60 m)'] = int(hi.sum()); s = s[~hi]

# 4 flat-line runs of three hours or more of identical values
run_id = (s.diff() != 0).cumsum()
rs = s.groupby(run_id).transform(lambda x: (x.index[-1] - x.index[0]).total_seconds()/3600)
fl = rs >= 3; log['flatline_values_removed(run>=3h)'] = int(fl.sum()); s = s[~fl]

# 5 Hampel-type spike filter, two passes: more than 0.25 m from a centred 70-minute median
for it in range(2):
    dev = s - s.rolling('70min', center=True, min_periods=3).median()
    sp = dev.abs() > 0.25
    log[f'spikes_removed_pass{it+1}'] = int(sp.sum())
    s = s[~sp]
log['clean_records'] = len(s)
s.to_pickle(OUT / 'clean_10min.pkl')

# hourly values on the hour, by linear interpolation between neighbours no more
# than 30 minutes apart; anything wider is left missing
grid = pd.date_range(s.index[0].ceil('h'), s.index[-1].floor('h'), freq='h')
ts = s.index.as_unit('ns').asi8; tg = grid.as_unit('ns').asi8
idx = np.clip(np.searchsorted(ts, tg), 1, len(ts)-1)
t0, t1 = ts[idx-1], ts[idx]; y0, y1 = s.values[idx-1], s.values[idx]
exact = ts[idx] == tg
wgt = np.where(t1 > t0, (tg - t0)/(t1 - t0 + 0.0), 0)
val = np.where(exact, y1, y0 + wgt*(y1 - y0))
ok = ((t1 - t0) <= 30*60*1e9) | exact
hourly = pd.Series(np.where(ok, val, np.nan), grid, name='h')
hourly.to_pickle(OUT / 'hourly.pkl')
log['hourly_grid_points'] = len(hourly)
log['hourly_missing'] = int(hourly.isna().sum())
log['hourly_missing_pct'] = round(100*hourly.isna().mean(), 2)
pd.Series(log).to_csv(RES / 'qc_log.csv')
print(pd.Series(log))
