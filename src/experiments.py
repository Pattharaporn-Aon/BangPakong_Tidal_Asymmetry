"""The 26 withheld-data experiments.

Days are removed from the fit in blocks, each covariance structure is refitted on
what remains, and the removed days are predicted. The design varies the size, the
position and the shape of what is withheld:

  A   one interior block, four sizes at two positions        =  8
  B   one block at the end, so the model predicts forward    =  4
  C1  the 75-day summer window, per year where complete      =  5
  C2  the 39-day winter window, per year where complete      =  5
  D   runs of one to ten days scattered through the record   =  4

The total of 26 follows from the design rather than from a target.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_d = pd.read_pickle(ROOT / 'data' / 'processed' / 'daily.pkl')
idx = _d.index
n = len(_d)
obs = ~np.isnan(_d.dmwl.values)

def cases():
    C = []
    for pct in (10, 20, 30, 40):
        L = int(round(n*pct/100))
        for cpos in (0.35, 0.65):
            s = int(n*cpos - L/2); m = np.zeros(n, bool); m[s:s+L] = True
            C.append(dict(group='A_interior_block',
                          label=f'{pct}% block centred {int(cpos*100)}%', pct=pct, mask=m))
    for pct in (10, 20, 30, 40):
        L = int(round(n*pct/100)); m = np.zeros(n, bool); m[n-L:] = True
        C.append(dict(group='B_end_prediction', label=f'last {pct}%', pct=pct, mask=m))
    for yr in (2020, 2022, 2023, 2024, 2025):
        m = (idx >= f'{yr}-06-25') & (idx <= f'{yr}-09-07')
        C.append(dict(group='C_realistic_JunSep_75d',
                      label=f'{yr}-06-25 to {yr}-09-07', pct=None, mask=np.asarray(m)))
    for yr in (2021, 2022, 2023, 2024, 2025):
        m = (idx >= f'{yr}-11-27') & (idx <= f'{yr+1}-01-04')
        C.append(dict(group='C_realistic_NovJan_39d',
                      label=f'{yr}-11-27 to {yr+1}-01-04', pct=None, mask=np.asarray(m)))
    rng = np.random.default_rng(2024)
    for pct in (10, 20, 30, 40):
        m = np.zeros(n, bool)
        while m[obs].mean() < pct/100:
            L = rng.integers(1, 11); s = rng.integers(0, n-L); m[s:s+L] = True
        C.append(dict(group='D_scattered_1to10d',
                      label=f'{pct}% scattered 1-10 d gaps', pct=pct, mask=m))
    return C

if __name__ == '__main__':
    C = cases()
    print(len(C), 'experiments')
    print(pd.Series([c['group'] for c in C]).value_counts().to_string())
