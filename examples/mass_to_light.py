import numpy as np
from imf import imf
import pylab as pl
from astropy.utils.console import ProgressBar

clusters, luminosities, masses, mean_luminosities, mean_masses, max_masses = {},{},{},{},{},{}
# uniform random sampling from 100 to 10^5 msun
for clmass in ProgressBar(np.concatenate([10**(np.random.rand(int(1e3))*1 + 4), 10**(np.random.rand(int(1e4))*2.5+1.5)])):
    clusters[clmass] = imf.make_cluster(clmass, 'kroupa', mmax=150, silent=True)
    # cluster luminosities
    luminosities[clmass] = imf.lum_of_cluster(clusters[clmass])
    masses[clmass] = clmass
    mean_luminosities[clmass] = np.mean(luminosities[clmass])
    mean_masses[clmass] = np.mean(clusters[clmass])
    max_masses[clmass] = np.max(clusters[clmass])


mass_to_light = np.array([mean_masses[k]/mean_luminosities[k] for k in sorted(clusters.keys())])

pl.figure(2).clf()
pl.semilogx(sorted(clusters.keys()), mass_to_light**-1, '.', alpha=0.1)
pl.xlabel("Cluster Mass")
pl.ylabel("Light to Mass $L_\odot / M_\odot$")
pl.ylim(6,21)
pl.savefig("light_to_mass_vs_mass.pdf")


pl.figure(3).clf()
pl.loglog(max_masses.keys(), max_masses.values(), '.', alpha=0.1)
pl.xlabel("Cluster Mass")
pl.ylabel("Maximum stellar mass")
pl.savefig("maxmass_vs_clustermass.pdf")

#pl.loglog(clusters.keys(), np.array(list(mean_luminosities.values())) / np.array(list(mean_masses.values())), '.', alpha=0.1)

# compute mass-to-light ratio vs age
