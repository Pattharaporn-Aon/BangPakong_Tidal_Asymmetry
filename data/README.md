# Data

## `raw/Clean_BangPaKong_2.xlsx`

Ten-minute water level at the Marine Department station on the Bang Pakong
estuary, Chachoengsao Province, Thailand, from 7 May 2019 to 23 June 2026.
353,644 rows, two columns: `DateTime` and `WaterLevel(m)`. Levels are in metres
above station datum.

The records were provided by the Marine Department of Thailand. Please cite the
Department as the source of the observations, and this repository for the
processing and analysis.

## `processed/` (not in version control)

Everything here is regenerated from the raw file and is therefore ignored by
git. Run, in order:

    python src/prep.py     # clean_10min.pkl, hourly.pkl   (about 1 minute)
    python src/daily.py    # daily.pkl, tide_coef_all.npy  (about 20 seconds)

`src/gp7_full_record.py` additionally writes `loo_*.npy`, the leave-one-out
residual of each covariance structure.
