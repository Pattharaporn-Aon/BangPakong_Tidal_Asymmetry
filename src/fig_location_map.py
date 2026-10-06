"""Location map of the Bang Pakong tide gauge (Fig. 1).

Reads  data/map/gebco_2026_inner_gulf.nc  (GEBCO 2026 grid, 12.4-14.0 N, 99.7-101.6 E)
       data/map/naturalearth/*.shp        (Natural Earth 1:10m and 1:50m; downloaded
                                           automatically on the first run)
Writes figures/fig_location_map.png and figures/fig_location_map.pdf

Coastlines, islands and rivers come from Natural Earth (public domain,
https://www.naturalearthdata.com). Bathymetry comes from the GEBCO 2026 grid
(https://www.gebco.net). The gauge position is from the Marine Department
station page (MD14), 13.485126 N, 101.002875 E.

Runtime under a minute after the download.
"""
import io, zipfile, urllib.request
from pathlib import Path
import numpy as np
import shapefile                     # pyshp
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPoly, Rectangle
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'data' / 'map'
NE_DIR = MAP / 'naturalearth'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]

NE_FILES = {'ne_10m_land': '10m/physical', 'ne_10m_minor_islands': '10m/physical',
            'ne_10m_rivers_lake_centerlines': '10m/physical', 'ne_50m_land': '50m/physical'}
NE_URL = 'https://naciscdn.org/naturalearth/{}/{}.zip'


def get_naturalearth():
    """Download and unpack the four Natural Earth layers if they are not present."""
    NE_DIR.mkdir(parents=True, exist_ok=True)
    for name, sub in NE_FILES.items():
        if (NE_DIR / f'{name}.shp').exists():
            continue
        print('downloading', name)
        with urllib.request.urlopen(NE_URL.format(sub, name)) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(NE_DIR)


get_naturalearth()
NE = str(NE_DIR) + '/'

GAUGE = (101.002875, 13.485126)

LAND = '#e4e2dc'
SEA = '#f4f7f9'
COAST = '#5f5e5a'
INK = '#2b2b2a'
MUTED = '#6b6a66'
MARK = '#c0392b'


def polys(path, bbox):
    """Return land polygons (lists of rings) intersecting bbox = (x0, y0, x1, y1)."""
    out = []
    for shp in shapefile.Reader(path).shapes():
        b = shp.bbox
        if b[2] < bbox[0] or b[0] > bbox[2] or b[3] < bbox[1] or b[1] > bbox[3]:
            continue
        parts = list(shp.parts) + [len(shp.points)]
        for i in range(len(parts) - 1):
            out.append(np.array(shp.points[parts[i]:parts[i + 1]]))
    return out


def draw_land(ax, path, bbox, lw):
    rings = polys(path, bbox)
    pc = PatchCollection([MplPoly(r, closed=True) for r in rings],
                         facecolor=LAND, edgecolor=COAST, linewidth=lw, zorder=2)
    ax.add_collection(pc)


def lines(ax, path, bbox, names, **kw):
    r = shapefile.Reader(path)
    for sr in r.iterShapeRecords():
        if sr.record['name'] not in names:
            continue
        pts = np.array(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i]:parts[i + 1]]
            ax.plot(seg[:, 0], seg[:, 1], **kw)


def setup(ax, bbox):
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1 / np.cos(np.deg2rad((bbox[1] + bbox[3]) / 2)))
    ax.set_facecolor(SEA)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)


def deg(v, ns):
    h = ('N' if v >= 0 else 'S') if ns else ('E' if v >= 0 else 'W')
    return f'{abs(v):.1f}°{h}' if v % 1 else f'{abs(v):.0f}°{h}'


plt.rcParams['font.family'] = 'DejaVu Serif'
fig = plt.figure(figsize=(7.0, 4.6))

# ---------- (b) inner Gulf of Thailand ----------
B = (99.75, 12.45, 101.55, 13.95)
ax = fig.add_axes([0.40, 0.08, 0.53, 0.88])
setup(ax, B)

# bathymetry: GEBCO 2026 15-arc-second grid
g = netCDF4.Dataset(MAP / 'gebco_2026_inner_gulf.nc')
glat, glon = g['lat'][:], g['lon'][:]
dep = -np.asarray(g['elevation'][:], dtype=float)
dep[dep <= 0] = np.nan
LEV = [0, 5, 10, 15, 20, 25, 30, 40, 60]
cmap = plt.get_cmap('Blues')
cols = [cmap(0.12 + 0.8 * i / (len(LEV) - 2)) for i in range(len(LEV) - 1)]
cf = ax.contourf(glon, glat, dep, levels=LEV, colors=cols, zorder=1)
ax.contour(glon, glat, dep, levels=[10, 20, 30], colors='white', linewidths=0.4, zorder=1.5)

draw_land(ax, NE + 'ne_10m_land', B, 0.6)
draw_land(ax, NE + 'ne_10m_minor_islands', B, 0.4)
lines(ax, NE + 'ne_10m_rivers_lake_centerlines', B, {'Chao Phraya', 'Mae Klong'},
      color='#7a9bb5', lw=0.8, zorder=3)

places = {'Bangkok': (100.515, 13.752), 'Chachoengsao': (101.076, 13.679),
          'Chon Buri': (101.0, 13.4), 'Si Racha': (100.929, 13.159),
          'Samut Prakan': (100.611, 13.607), 'Samut Sakhon': (100.274, 13.536)}
off = {'Bangkok': (0.03, 0.03), 'Chachoengsao': (0.04, 0.02), 'Chon Buri': (0.05, -0.02),
       'Si Racha': (0.05, -0.02), 'Samut Prakan': (0.04, 0.0), 'Samut Sakhon': (-0.05, 0.05)}
ha = {'Samut Sakhon': 'right'}
for n, (x, y) in places.items():
    ax.plot(x, y, 's', ms=2.6, color=INK, zorder=4)
    dx, dy = off[n]
    ax.text(x + dx, y + dy, n, fontsize=6.5, color=INK, ha=ha.get(n, 'left'), va='center', zorder=5)

ax.plot(*GAUGE, '^', ms=8, color=MARK, mec='white', mew=0.8, zorder=6)
ax.annotate('Bang Pakong gauge\n13.485°N, 101.003°E', xy=GAUGE, xytext=(100.62, 13.30),
            fontsize=7, color=INK, path_effects=HALO, ha='center', va='center', zorder=6,
            arrowprops=dict(arrowstyle='-', color=INK, lw=0.6, shrinkB=5))
ax.text(101.12, 13.56, 'Bang Pakong\nRiver', fontsize=6.5, style='italic', color='#3d6585', ha='left')
ax.text(100.12, 13.86, 'Chao Phraya\nRiver', fontsize=6.5, style='italic', color='#3d6585', ha='center')
ax.text(100.30, 12.95, 'Inner Gulf\nof Thailand', fontsize=9, style='italic', color=INK, ha='center', path_effects=HALO)
ax.text(99.95, 13.10, 'THAILAND', fontsize=7, color=MUTED, ha='center', alpha=0.9)

ax.set_xticks([100.0, 100.5, 101.0, 101.5]); ax.set_yticks([12.5, 13.0, 13.5])
ax.yaxis.tick_right()
ax.set_xticklabels([deg(v, False) for v in ax.get_xticks()])
ax.set_yticklabels([deg(v, True) for v in ax.get_yticks()])

# scale bar 20 km at 12.6 N
km = 20 / (111.32 * np.cos(np.deg2rad(12.6)))
x0, y0 = 101.05, 12.58
ax.plot([x0, x0 + km], [y0, y0], color=INK, lw=1.6, solid_capstyle='butt', zorder=6)
ax.text(x0 + km / 2, y0 + 0.035, '20 km', fontsize=6.5, ha='center', color=INK, path_effects=HALO)
# north arrow
ax.annotate('', xy=(101.45, 13.88), xytext=(101.45, 13.74),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9))
ax.text(101.45, 13.90, 'N', fontsize=8, ha='center', va='bottom', color=INK)
ax.text(0.015, 0.985, '(b)', transform=ax.transAxes, fontsize=9, va='top', fontweight='bold')

# ---------- (a) regional overview ----------
A = (92.0, -1.0, 112.0, 24.0)
ia = fig.add_axes([0.075, 0.30, 0.29, 0.62])
setup(ia, A)
draw_land(ia, NE + 'ne_50m_land', A, 0.4)
ia.add_patch(Rectangle((B[0], B[1]), B[2] - B[0], B[3] - B[1], fill=False, ec=MARK, lw=1.0, zorder=5))
ia.text(102.5, 9.0, 'Gulf of\nThailand', fontsize=6, style='italic', color=MUTED, ha='center')
ia.text(109.3, 16.5, 'South\nChina\nSea', fontsize=6, style='italic', color=MUTED, ha='center')
ia.text(95.0, 13.0, 'Andaman\nSea', fontsize=6, style='italic', color=MUTED, ha='center')
ia.set_xticks([95, 105]); ia.set_yticks([0, 10, 20])
ia.set_xticklabels([deg(v, False) for v in ia.get_xticks()])
ia.set_yticklabels([deg(v, True) for v in ia.get_yticks()])
ia.text(0.04, 0.975, '(a)', transform=ia.transAxes, fontsize=9, va='top', fontweight='bold')

cax = fig.add_axes([0.075, 0.215, 0.29, 0.022])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=LEV)
cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
cb.outline.set_linewidth(0.5)
cb.set_label('Water depth (m)', fontsize=7, color=INK)
fig.text(0.075, 0.06, 'Coastlines: Natural Earth 1:10m.\nBathymetry: GEBCO 2026 grid.\nGauge: Marine Department station MD14.',
         fontsize=6, color=MUTED, va='bottom')

for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'fig_location_map.{ext}', dpi=600)
print('written to figures/fig_location_map.png and .pdf')
