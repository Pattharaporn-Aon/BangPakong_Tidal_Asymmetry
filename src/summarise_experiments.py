"""Summarise the 26 withheld-data experiments.

Reads  results/withheld_experiments.csv
Writes results/withheld_group_means.csv, results/withheld_paired_tests.csv,
       results/withheld_coverage95.csv

The unit of replication is the experiment, not the day: 26 cases, not 10276
days. Pairing is by case against GP-2M-SA.
"""
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
d = pd.read_csv(RES / 'withheld_experiments.csv')
VAR = ['GPM12', 'GP', 'GPM52', 'GPRQ', 'GP2M32', 'GP2M32SA', 'GPQP']
LAB = dict(zip(d.model, d.kernel))
GN = {'A_interior_block':'A', 'B_end_prediction':'B', 'C_realistic_JunSep_75d':'C1',
      'C_realistic_NovJan_39d':'C2', 'D_scattered_1to10d':'D'}
d['g'] = d.group.map(lambda x: GN.get(x, x))

P = d.pivot(index='case', columns='model', values='RMSE_cm')[VAR]
G = d.groupby('case').g.first()
gm = P.groupby(G).mean(); gm.loc[f'all {len(P)}'] = P.mean()
gm.columns = [LAB[k] for k in gm.columns]; gm.index.name = 'group'
gm.round(3).to_csv(RES / 'withheld_group_means.csv')
print('mean RMSE (cm) over the withheld days'); print(gm.round(3).to_string())

ref = P['GP2M32SA']; rows = []
for k in VAR:
    if k == 'GP2M32SA': continue
    dd = (P[k] - ref).values
    t, p_t = stats.ttest_rel(P[k], ref); _, p_w = stats.wilcoxon(dd)
    rows.append(dict(kernel=LAB[k], delta_cm=dd.mean(), t=t, p_t=p_t, p_wilcoxon=p_w,
                     sa_better_in=int((dd > 0).sum()), n_cases=len(dd)))
pt = pd.DataFrame(rows)
pt.round(4).to_csv(RES / 'withheld_paired_tests.csv', index=False)
print('\npaired comparison with GP-2M-SA'); print(pt.round(3).to_string(index=False))
print('delta is the mean difference in RMSE; positive means GP-2M-SA is the better.')

C = d.pivot(index='case', columns='model', values='cover95')[VAR]
cov = C.groupby(G).mean(); cov.loc[f'all {len(C)}'] = C.mean()
cov.columns = [LAB[k] for k in cov.columns]; cov.index.name = 'group'
cov.round(2).to_csv(RES / 'withheld_coverage95.csv')
print('\ncoverage of the nominal 95 % interval (%)'); print(cov.round(1).to_string())
print('\nbest structure in each case')
print(P.idxmin(axis=1).map(LAB).value_counts().to_string())
