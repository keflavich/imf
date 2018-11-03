import numpy as np
import imf
import pylab as pl
from astroquery.vizier import Vizier
from labellines import labelLine, labelLines

pl.rcParams['font.size'] = 18

Vizier.ROW_LIMIT=1e7

tbl = Vizier.get_catalogs('J/A+A/537/A146/iso')[0]

agemass = {}
agelum = {}
for age in np.unique(tbl['logAge']):
    agemass[age] = tbl[tbl['logAge']==age]['Mass'].max()
    agelum[age] = tbl[tbl['logAge']==age]['logL'].max()


# hr diagram
pl.figure(2, figsize=(8,8)).clf()

ok = tbl['logAge'] == 6.5

subtbl = tbl[ok]
subtbl.sort('Mass')
lowmass = subtbl[subtbl['Mass'] < 2]

subtbl = subtbl[(subtbl['Mass'] < 25) & (subtbl['logTe']<4.6)]
highmass = subtbl[(subtbl['Mass'] > 20) & (subtbl['logTe']<4.6)]
subtbl=subtbl[::20]

# fill in low-mass
Lfit = np.polyfit(np.log10(lowmass['Mass']), lowmass['logL'], 1)
# not exactly right; there's a break at 0.43ish
masses = np.logspace(np.log10(0.1), np.log10(0.8))
lums = np.poly1d(Lfit)(np.log10(masses))
#lums[masses<(0.43)] = np.log10(0.23*(masses[masses<(0.43)])**2.3)
Tfit = np.polyfit(np.log10(lowmass['Mass']), lowmass['logTe'], 1)
tems = np.poly1d(Tfit)(np.log10(masses))

hmasses = np.logspace(np.log10(25), np.log10(60),5)
Lfit = np.polyfit(np.log10(highmass['Mass']), highmass['logL'], 1)
hlums = np.poly1d(Lfit)(np.log10(hmasses))
#lums[masses<(0.43)] = np.log10(0.23*(masses[masses<(0.43)])**2.3)
Tfit = np.polyfit(np.log10(highmass['Mass']), highmass['logTe'], 1)
htems = np.poly1d(Tfit)(np.log10(hmasses))

colors = [imf.color_from_mass(m) for m in subtbl['Mass']]
#pl.gca().set_xscale('log')
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

colors = [imf.color_from_mass(m) for m in hmasses]
pl.scatter(10**htems,
           hmasses,
           c=colors,
           s=(10**hlums)**0.25*45)

lines = []
for age in (6.5, 7, 8, 10):
    L, = pl.plot([10**tems.min(), (10**htems.max())],
                 [agemass[age]]*2, linestyle='--', color='k',
                 label="$10^{{{0}}}$ yr".format(age))
    lines.append(L)

labelLines(lines)
pl.xlabel("Temperature")
pl.ylabel("Mass")
pl.tight_layout()
pl.savefig("mass_lum_diagram.svg")#, bbox_inches='tight')

pl.figure(3, figsize=(8,8)).clf()

colors = [imf.color_from_mass(m) for m in subtbl['Mass']]
#pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.scatter(10**subtbl['logTe'],
           10**subtbl['logL'],
           c=colors,
           s=(subtbl['Mass'])*5)

colors = [imf.color_from_mass(m) for m in masses]
pl.scatter(10**tems,
           10**lums,
           c=colors,
           s=masses*5)

colors = [imf.color_from_mass(m) for m in hmasses]
pl.scatter(10**htems,
           10**hlums,
           c=colors,
           s=hmasses*5)

lines = []
for age in (6.5, 7, 8, 10):
    L, = pl.plot([10**tems.min(), (10**htems.max())],
                 [10**agelum[age]]*2,
                 linestyle='--', color='k',
                 label="$10^{{{0}}}$ yr".format(age))
    lines.append(L)

labelLines(lines)
pl.xlabel("Temperature")
pl.ylabel("Luminosity")
pl.tight_layout()
pl.savefig("HR_diagram.svg")#, bbox_inches='tight')
