import numpy as np

from .imf import make_cluster
from .lum import lum_of_star

def color_from_mass(mass, outtype=float):
    """
    Returns a color given a stellar mass. Colors are interpolated
    between RGB values sourced from `vendian.org <http://vendian.org/>`_.
    ``outtype`` can be either ``int`` or ``float`` (default = ``float``),
    which determines the type of the output color values.
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
    default is ``color_from_mass``.
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
    Returns information necessary to visualize a cluster; stellar 
    masses, y-axis positions, and colors. If existing sampled masses 
    are provided, those will be used; otherwise, a cluster meeting
    the specifications is created.
    
    Parameters
    ----------
    masses: list/array
        The masses of an existing sampled cluster in solar masses. 
        Either ``masses`` or ``mtot`` must be provided.
    mtot: float
        The mass of the cluster in solar masses. Either ``mtot`` or ``masses``
        must be provided.
    massfunc: MassFunction
        The mass function to use for sampling/positioning
    log: bool
        Whether the y-axis is log-scaled (default = True)
        
    Returns
    -------
    cluster: array
        The array of stellar masses that makes up the cluster
    yax: array
        The array of y values associated with the stellar masses
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
