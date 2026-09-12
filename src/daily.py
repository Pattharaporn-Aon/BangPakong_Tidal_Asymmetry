"""Daily mean water level, formed from the tidal residual.

Reads  data/processed/hourly.pkl
Writes data/processed/daily.pkl, results/daily_gaps.csv

On a day with all twenty-four hours present a plain average and this one are the
same. On an incomplete day a plain average is biased, because the hours that are
missing are not spread evenly over the tidal cycle; at this station that bias
exceeds 0.3 m on the worst days. The harmonic prediction is therefore removed
first, the residual is averaged over the day, and the mean level Z0 is added
back. A day is kept only if at least eighteen hours survive quality control.
"""
import numpy as np, pandas as pd
from pathlib import Path
import tide

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES  = ROOT / 'results'

h = pd.read_pickle(PROC / 'hourly.pkl'); ho = h.dropna()
names = list(tide.CONST)
coef = tide.fit(ho.index, ho.values, names)
np.save(PROC / 'tide_coef_all.npy', coef)

res  = ho - tide.predict(ho.index, names, coef, with_mean=True)
cnt  = res.resample('D').count()
full = pd.date_range(h.index[0].normalize(), h.index[-1].normalize(), freq='D')
dm   = (res.resample('D').mean() + coef[0]).reindex(full)
cnt  = cnt.reindex(full).fillna(0).astype(int)
daily = pd.DataFrame({'dmwl': dm.where(cnt >= 18), 'hours': cnt})
daily.index.name = 'date'
daily.to_pickle(PROC / 'daily.pkl')

print(f'calendar days {len(daily)}, kept {int(daily.dmwl.notna().sum())}, '
      f'dropped {int(daily.dmwl.isna().sum())}')
m = daily.dmwl.isna()
blk = (m != m.shift()).cumsum()[m]
gaps = (daily[m].groupby(blk)
        .apply(lambda d: pd.Series({'start': d.index[0].date(),
                                    'end': d.index[-1].date(), 'days': len(d)}))
        .sort_values('days', ascending=False).reset_index(drop=True))
gaps.to_csv(RES / 'daily_gaps.csv', index=False)
print(gaps.head(6).to_string())
