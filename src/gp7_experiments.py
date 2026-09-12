"""Run the seven covariance structures over the 26 withheld-data experiments.

Writes results/withheld_experiments.csv

This is the slow step: seven fits per case at O(n^3), roughly six minutes each,
so about three hours in total. Cases may be given on the command line to split
the work over several processes, for example

    python src/gp7_experiments.py 0 1 2 3 4 5 6 7 8 9 10 11 12
    python src/gp7_experiments.py 13 14 15 16 17 18 19 20 21 22 23 24 25
"""
import numpy as np, pandas as pd, sys, time
from pathlib import Path
from models import GPModel, tyears
import experiments as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'cases'; OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_pickle(ROOT / 'data' / 'processed' / 'daily.pkl')
t_all = tyears(d.index); y_all = d.dmwl.values
obs = ~np.isnan(y_all)
VAR = ['GPM12', 'GP', 'GPM52', 'GPRQ', 'GP2M32', 'GP2M32SA', 'GPQP']
LAB = {'GPM12':'Matern 1/2','GP':'Matern 3/2','GPM52':'Matern 5/2','GPRQ':'rational quadratic',
       'GP2M32':'two-scale M3/2','GP2M32SA':'GP-2M-SA','GPQP':'M3/2 + quasi-periodic'}
C = E.cases()
todo = [int(a) for a in sys.argv[1:]] or range(len(C))
for ci in todo:
    f = OUT / f'case_{ci:02d}.csv'
    if f.exists(): continue
    c = C[ci]; test = c['mask'] & obs; train = obs & ~c['mask']
    tt, yt = t_all[train], y_all[train]; ts, ys = t_all[test], y_all[test]
    rows = []; t0 = time.time()
    for k in VAR:
        g = GPModel(k).fit(tt, yt)
        mu, sd = g.predict(ts)
        sdy = np.sqrt(sd**2 + g.params['sn2']); e = mu - ys
        rows.append(dict(case=ci, group=c['group'], label=c['label'], model=k, kernel=LAB[k],
                         RMSE_cm=100*np.sqrt(np.mean(e**2)), MAE_cm=100*np.mean(np.abs(e)),
                         bias_cm=100*np.mean(e), cover95=100*np.mean(np.abs(e) <= 1.96*sdy),
                         n_test=int(test.sum()), n_train=int(train.sum())))
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f'case {ci:2d} {c["group"][:14]:14s} {time.time()-t0:6.1f}s  ' +
          '  '.join(f'{r["model"]}:{r["RMSE_cm"]:.2f}' for r in rows), flush=True)

parts = sorted(OUT.glob('case_*.csv'))
if len(parts) == len(C):
    pd.concat([pd.read_csv(p) for p in parts]).to_csv(
        ROOT / 'results' / 'withheld_experiments.csv', index=False)
    print(f'\nall {len(C)} cases complete, merged into results/withheld_experiments.csv')
else:
    print(f'\n{len(parts)} of {len(C)} cases done so far')
