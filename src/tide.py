import numpy as np, pandas as pd
# Constituent speeds (deg/hour) and nodal type
CONST = {
 '2Q1':(12.8542862,'O1'),'Q1':(13.3986609,'O1'),'RHO1':(13.4715145,'O1'),'O1':(13.9430356,'O1'),
 'NO1':(14.4966939,'M1'),'P1':(14.9589314,None),'S1':(15.0,None),'K1':(15.0410686,'K1'),
 'J1':(15.5854433,'J1'),'OO1':(16.1391017,'OO1'),
 '2N2':(27.8953548,'M2'),'MU2':(27.9682084,'M2'),'N2':(28.4397295,'M2'),'NU2':(28.5125831,'M2'),
 'M2':(28.9841042,'M2'),'LAM2':(29.4556253,'M2'),'L2':(29.5284789,'M2'),'T2':(29.9589333,None),
 'S2':(30.0,None),'K2':(30.0821373,'K2'),'2SM2':(31.0158958,'M2'),
 'MO3':(42.9271398,'MO3'),'M3':(43.4761563,'M3'),'MK3':(44.0251729,'MK3'),'SK3':(45.0410686,'K1'),
 'MN4':(57.4238337,'M4'),'M4':(57.9682084,'M4'),'MS4':(58.9841042,'M2'),'MK4':(59.0662415,'MK3'),
 'S4':(60.0,None),'M6':(86.9523127,'M6'),'2MS6':(87.9682084,'M4'),'M8':(115.9364166,'M8'),
}
def nodal(t):
    """f,u (u in radians) per nodal type; t = pandas DatetimeIndex"""
    jd = t.to_julian_date().values
    T = (jd-2451545.0)/36525.0
    N = np.deg2rad(125.04452 - 1934.136261*T)
    c, s, c2, s2 = np.cos(N), np.sin(N), np.cos(2*N), np.sin(2*N)
    fM2 = 1.0004-0.0373*c+0.0002*c2; uM2 = np.deg2rad(-2.14*s)
    fO1 = 1.0089+0.1871*c-0.0147*c2; uO1 = np.deg2rad(10.80*s-1.34*s2)
    fK1 = 1.0060+0.1150*c-0.0088*c2; uK1 = np.deg2rad(-8.86*s+0.68*s2)
    fK2 = 1.0246+0.2863*c+0.0083*c2; uK2 = np.deg2rad(-17.74*s+0.68*s2)
    fJ1 = 1.0129+0.1676*c-0.0170*c2; uJ1 = np.deg2rad(-12.94*s+1.34*s2)
    fOO1= 1.1027+0.6504*c+0.0317*c2; uOO1= np.deg2rad(-36.68*s+4.02*s2)
    one = np.ones_like(c); zero = np.zeros_like(c)
    return {None:(one,zero),'M2':(fM2,uM2),'O1':(fO1,uO1),'K1':(fK1,uK1),'K2':(fK2,uK2),
            'J1':(fJ1,uJ1),'OO1':(fOO1,uOO1),'M1':(one,zero),
            'M4':(fM2**2,2*uM2),'M6':(fM2**3,3*uM2),'M8':(fM2**4,4*uM2),
            'MK3':(fM2*fK1,uM2+uK1),'MO3':(fM2*fO1,uM2+uO1),'M3':(fM2**1.5,1.5*uM2)}
T0 = pd.Timestamp('2020-01-01')
def design(t, names):
    th = ((t - T0).total_seconds().values/3600.0)
    nd = nodal(t)
    cols = []
    for n in names:
        w, typ = CONST[n]; f,u = nd[typ]
        arg = np.deg2rad(w)*th + u
        cols += [f*np.cos(arg), f*np.sin(arg)]
    return np.column_stack(cols)
def fit(t, y, names, w=None):
    X = design(t, names)
    X = np.column_stack([np.ones(len(t)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef
def predict(t, names, coef, with_mean=False):
    X = design(t, names)
    return (coef[0] if with_mean else 0.0) + X@coef[1:]
def amplitudes(names, coef):
    a = coef[1::2]; b = coef[2::2]
    return pd.DataFrame({'constituent':names,'speed_deg_per_h':[CONST[n][0] for n in names],
                         'amplitude_m':np.hypot(a,b),'phase_deg_rel2020':np.rad2deg(np.arctan2(b,a))%360})

# Constituents given an annual modulation. K1, S2 and P1 are deliberately absent:
# modulating them at one cycle per year puts a sideband exactly on S1, T2 and S1,
# which are already in the set, and the design matrix becomes singular.
MOD_NAMES = ('M2','N2','O1','Q1','K2','M4','MS4','MN4','MK3','MO3','M6','2MS6')

def design_mod(t, names, mod_names=MOD_NAMES):
    """Harmonic design with annual modulation (x [sin, cos] of annual cycle) of main constituents."""
    X = design(t, names)
    ty = ((t - T0).total_seconds().values/86400.0)/365.25
    sa, ca = np.sin(2*np.pi*ty), np.cos(2*np.pi*ty)
    extra = []
    for n in mod_names:
        j = names.index(n); c, s = X[:,2*j], X[:,2*j+1]
        extra += [c*sa, c*ca, s*sa, s*ca]
    return np.column_stack([X]+extra)
def fit_mod(t, y, names):
    X = np.column_stack([np.ones(len(t)), design_mod(t, names)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); return coef
def predict_mod(t, names, coef, with_mean=False):
    return (coef[0] if with_mean else 0.0) + design_mod(t, names)@coef[1:]
