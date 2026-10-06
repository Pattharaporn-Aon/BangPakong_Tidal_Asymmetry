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

## `map/gebco_2026_inner_gulf.nc`

Water depth for the location map, cut from the GEBCO 2026 global grid
(15 arc-second) for 12.4-14.0 N and 99.7-101.6 E. Variables `lat`, `lon` and
`elevation` (metres, negative below sea level). Source: GEBCO Compilation Group
(2026), https://www.gebco.net. Redistributed with attribution under the GEBCO
terms of use.

## `map/naturalearth/` (not in version control)

Natural Earth 1:10m land, minor islands and rivers, and 1:50m land, downloaded
by `src/fig_location_map.py` on its first run. Public domain,
https://www.naturalearthdata.com.
