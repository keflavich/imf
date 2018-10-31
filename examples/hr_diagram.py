import numpy as np
import imf
import pylab as pl
from astroquery.vizier import Vizier
from labellines import labelLine, labelLines


Vizier.ROW_LIMIT=1e7

tbl = Vizier.get_catalogs('J/A+A/537/A146/iso')[0]

agemass = {}
for age in np.unique(tbl['logAge']):
    agemass[age] = tbl[tbl['logAge']==age]['Mass'].max()


# hr diagram
pl.figure(2).clf()

ok = tbl['logAge'] == 6.5

subtbl = tbl[ok]
subtbl.sort('Mass')
lowmass = subtbl[subtbl['Mass'] < 2]

subtbl=subtbl[::10]
subtbl = subtbl[subtbl['Mass'] < 30]

# fill in low-mass
Lfit = np.polyfit(np.log10(lowmass['Mass']), lowmass['logL'], 1)
masses = np.logspace(np.log10(0.03), np.log10(0.8))
lums = np.poly1d(Lfit)(np.log10(masses))
Tfit = np.polyfit(np.log10(lowmass['Mass']), lowmass['logTe'], 1)
tems = np.poly1d(Tfit)(np.log10(masses))

colors = [imf.color_from_mass(m) for m in subtbl['Mass']]
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.scatter(10**subtbl['logTe'],
           subtbl['Mass'],
           c=colors,
           s=(10**subtbl['logL'])**0.25*45)

colors = [imf.color_from_mass(m) for m in masses]
pl.scatter(10**tems,
           masses,
           c=colors,
           s=(10**lums)**0.25*45)

lines = []
for age in (6.5, 7, 8, 9):
    L, = pl.plot([10**tems.min(), (10**subtbl['logTe'].max())], [agemass[age]]*2, linestyle='--', color='k',
                 label="$10^{{{0}}}$ yr".format(age))
    lines.append(L)

labelLines(lines)
