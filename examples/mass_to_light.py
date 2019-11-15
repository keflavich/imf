import numpy as np
import os
import json
from imf import imf
import pylab as pl
import matplotlib
from astropy.utils.console import ProgressBar

pl.rc('font', size=16)

if os.path.exists('synth_data_m_to_l.json'):
    with open('synth_data_m_to_l.json', 'r') as fh:
        synth_data = json.load(fh)
else:
    synth_data = {}

# uniform random sampling from 100 to 10^5 msun
for stop_crit in ('nearest', 'before', 'after', 'sorted'):
    print(stop_crit)
    if stop_crit not in synth_data:
        clusters, luminosities, masses, mean_luminosities, mean_masses, max_masses, number = {},{},{},{},{},{},{}
        for clmass in ProgressBar(np.concatenate([10**(np.random.rand(int(1e3))*1 + 4), 10**(np.random.rand(int(1e4))*2.5+1.5)])):
            key = str(clmass) # for jsonification
            clusters[key] = imf.make_cluster(clmass, 'kroupa', mmax=150, silent=True, stop_criterion=stop_crit)
            # cluster luminosities
            luminosities[key] = imf.lum_of_cluster(clusters[key])
            masses[key] = clmass
            number[key] = len(clusters[key])
            #mean_luminosities[clmass] = np.mean(luminosities[clmass])
            mean_masses[key] = np.mean(clusters[key])
            max_masses[key] = np.max(clusters[key])

        synth_data[stop_crit] = {#'clusters': clusters,
                                 'number': number,
                                 'luminosities': luminosities,
                                 'masses': masses,
                                 'mean_luminosities': mean_luminosities,
                                 'mean_masses': mean_masses,
                                 'max_masses': max_masses}
    else:
        max_masses = synth_data[stop_crit]['max_masses']
        #mean_luminosities = synth_data[stop_crit]['mean_luminosities']
        mean_masses = synth_data[stop_crit]['mean_masses']
        number = synth_data[stop_crit]['number']
        masses = synth_data[stop_crit]['masses']
        luminosities = synth_data[stop_crit]['luminosities']

    clmasses = sorted(map(float, masses))

    mass_to_light = np.array([k/10**luminosities[str(k)] for k in clmasses])

    # shouldn't this converge to a single value? (yes, don't divide linear by log)
    pl.figure(2).clf()
    pl.loglog(clmasses, mass_to_light**-1, '.', alpha=0.1)
    pl.xlabel("Cluster Mass")
    pl.ylabel("Light to Mass $L_\odot / M_\odot$")
    pl.ylim(1,5e4)
    pl.savefig(f"light_to_mass_vs_mass_{stop_crit}.png", bbox_inches='tight', dpi=200)
    pl.savefig(f"light_to_mass_vs_mass_{stop_crit}.pdf", bbox_inches='tight')


    pl.figure(3).clf()
    pl.loglog(list(map(float, max_masses.keys())), max_masses.values(), '.', alpha=0.1)
    pl.xlabel("Cluster Mass")
    pl.ylabel("Maximum stellar mass")
    pl.savefig(f"maxmass_vs_clustermass_{stop_crit}.png", bbox_inches='tight', dpi=200)
    pl.savefig(f"maxmass_vs_clustermass_{stop_crit}.pdf", bbox_inches='tight')

    m_to_ls = []
    slopes = np.linspace(1.7, 2.9, 20)
    for slope in ProgressBar(slopes):
        Kroupa = imf.Kroupa(p3=slope)
        cluster = imf.make_cluster(1e5, Kroupa, mmax=120, silent=True,
                                   stop_criterion=stop_crit)
        lum = imf.lum_of_cluster(cluster)
        m_to_l = 1e5/10**lum
        m_to_ls.append(m_to_l)

    pl.figure(4).clf()
    pl.plot(slopes, m_to_ls)
    pl.xlabel("Upper-end power-law slope $\\alpha$")
    pl.ylabel("Mass-to-light ratio [M$_\odot$/L$_\odot$]")
    ax = pl.gca()
    tw = ax.twinx()
    tw.set_ylim(m_to_ls[0]/m_to_ls[10], m_to_ls[-1]/m_to_ls[10])
    tw.set_ylabel("M/L / M/L($\\alpha=2.3$)")

    pl.savefig(f"masstolight_vs_slope_{stop_crit}.pdf", bbox_inches='tight')
    pl.savefig(f"masstolight_vs_slope_{stop_crit}.png", bbox_inches='tight', dpi=200)


    pl.figure(5).clf()
    pl.semilogy(slopes, m_to_ls)
    ax = pl.gca()
    ylim = ax.get_ylim()
    ax.vlines(2.3, 1e-5, 1, linestyle='--', color='k', alpha=0.2, zorder=-10)
    ax.set_ylim(ylim)
    pl.xlabel("Upper-end power-law slope $\\alpha$")
    pl.ylabel("Mass-to-light ratio [M$_\odot$/L$_\odot$]")
    tw = ax.twinx()
    tw.semilogy()
    tw.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    tw.set_yticks([0.5,1,2,5])
    tw.set_ylim(m_to_ls[0]/m_to_ls[10], m_to_ls[-1]/m_to_ls[10])
    tw.set_xlim(ax.get_xlim())
    tw.hlines(1, 1, 5, linestyle='--', color='k', alpha=0.2, zorder=-20)
    #tw.grid(which='major', linestyle='--')
    tw.set_ylabel("M/L / M/L($\\alpha=2.3$)")

    pl.savefig(f"masstolight_vs_slope_log_{stop_crit}.pdf", bbox_inches='tight')
    pl.savefig(f"masstolight_vs_slope_log_{stop_crit}.png", bbox_inches='tight', dpi=200)

    #pl.loglog(clusters.keys(), np.array(list(mean_luminosities.values())) / np.array(list(mean_masses.values())), '.', alpha=0.1)

    # compute mass-to-light ratio vs age

with open('synth_data_m_to_l.json', 'w') as fh:
    json.dump(synth_data, fh)
