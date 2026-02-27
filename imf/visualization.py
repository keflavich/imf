import numpy as np

from .imf import make_cluster
from .lum import lum_of_star

def color_from_mass(mass, outtype=float):
    """
    Use vendian.org colors:
    100 O2(V)        150 175 255   #9db4ff
     50 O5(V)        157 180 255   #9db4ff
     20 B1(V)        162 185 255   #a2b9ff
     10 B3(V)        167 188 255   #a7bcff
      8 B5(V)        170 191 255   #aabfff
      6 B8(V)        175 195 255   #afc3ff
    2.2 A1(V)        186 204 255   #baccff
    2.0 A3(V)        192 209 255   #c0d1ff
   1.86 A5(V)        202 216 255   #cad8ff
    1.6 F0(V)        228 232 255   #e4e8ff
    1.5 F2(V)        237 238 255   #edeeff
    1.3 F5(V)        251 248 255   #fbf8ff
    1.2 F8(V)        255 249 249   #fff9f9
      1 G2(V)        255 245 236   #fff5ec
   0.95 G5(V)        255 244 232   #fff4e8
   0.90 G8(V)        255 241 223   #fff1df
   0.85 K0(V)        255 235 209   #ffebd1
   0.70 K4(V)        255 215 174   #ffd7ae
   0.60 K7(V)        255 198 144   #ffc690
   0.50 M2(V)        255 190 127   #ffbe7f
   0.40 M4(V)        255 187 123   #ffbb7b
   0.35 M6(V)        255 187 123   #ffbb7b
   0.30 M8(V)        255 167 123   #ffbb7b  #added                                           
    """  

    mcolor = {100: (150, 175, 255),
              50: (157, 180, 255),
              20: (162, 185, 255),
              10: (167, 188, 255),
	       8: (170, 191, 255),
               6: (175, 195, 255),
             2.2: (186, 204, 255),
             2.0: (192, 209, 255),
            1.86: (202, 216, 255),
             1.6: (228, 232, 255),
             1.5: (237, 238, 255),
             1.3: (251, 248, 255),
             1.2: (255, 249, 249),
               1: (255, 245, 236),
            0.95: (255, 244, 232),
            0.90: (255, 241, 223),
            0.85: (255, 235, 209),
            0.70: (255, 215, 174),
            0.60: (255, 198, 144),
            0.50: (255, 190, 127),
            0.40: (255, 187, 123),
            0.35: (255, 187, 123),
            0.30: (255, 177, 113),
            0.20: (255, 107, 63),
            0.10: (155, 57, 33),
            0.10: (155, 57, 33),
           0.003: (105, 27, 0),
              }

    keys = sorted(mcolor.keys())

    reds, greens, blues = zip(*[mcolor[k] for k in keys])
    r = np.interp(mass, keys, reds)
    g = np.interp(mass, keys, greens)
    b = np.interp(mass, keys, blues)

    if outtype == int:
        return (r, g, b)
    elif outtype == float:
        return (r / 255., g / 255., b / 255.)
    else:
        raise NotImplementedError

def color_of_cluster(cluster, colorfunc=color_from_mass):
    """
    Returns the luminosity-weighted average color of a cluster in (r, g, b).
    Requires the input of a function which maps stellar mass to RGB color; 
    default is color_from_mass.
    """
    colors = np.array([colorfunc(m) for m in cluster])
    luminosities = 10**np.array([lum_of_star(m) for m in cluster])
    mean_color = (colors *
                  luminosities[:, None]).sum(axis=0) / luminosities.sum()
    return mean_color

def plotinfo(masses=None,
             mtot=None,
             massfunc=None,
             log=True,
             **kwargs):
    """
    This function returns information necessary to visualize a cluster;
    stellar masses, y-axis positions, and colors. If existing sampled 
    masses are provided, those will be used; otherwise, 'plot_cluster'
    creates a cluster using `make_cluster`. Once stars are sampled, each 
    star is assigned a color based on the vendian.org colors using 
    `color_from_mass` and a random Y-value  distributed underneath the 
    specified mass function's curve.
    
    Parameters
    ----------
    masses: list/array
        The masses of an existing sampled cluster in solar masses. 
        Either 'masses' or 'mtot' must be provided.
    mtot: float
        The mass of the cluster in solar masses. Either 'mtot' or 'masses'
        must be provided.
    massfunc: str
        A MassFunction instance
    log: bool
        Is the Y-axis log-scaled?
        
    Returns
    -------
    cluster: array
        The array of stellar masses that makes up the cluster
    yax: array
        The array of Y-values associated with the stellar masses
    colors: list
        A list of color tuples associated with each star
    """
    if masses is None:
        cluster = make_cluster(mtot,
                               massfunc=massfunc,
                               mmin=massfunc.mmin,
                               mmax=massfunc.mmax,
                               **kwargs)
    else:
        cluster = np.copy(masses)

    colors = [color_from_mass(m) for m in cluster]

    maxmass = cluster.max()
    pmin = massfunc(maxmass)
    if log:
        yax = [
            np.random.rand() * (np.log10(massfunc(m)) - np.log10(pmin)) +
            np.log10(pmin) for m in cluster
        ]
    else:
        yax = [
            np.random.rand() * ((massfunc(m)) / (pmin)) + (pmin)
            for m in cluster
        ]

    assert all(np.isfinite(yax))

    return cluster, yax, colors
