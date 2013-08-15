from imf import color_of_cluster,make_cluster,lum_of_cluster
import numpy as np

if __name__ == "__main__":
    import pylab as pl

    alpha = 2
    m0 = 5e2
    mmax = 5e5
    cluster_mass_xax = np.logspace(np.log10(m0),np.log10(mmax),1e4)
    def pr(m):
        return (m/m0)**-alpha
    probabilities = pr(cluster_mass_xax)
    cdf = probabilities.cumsum()
    cdf /= cdf.max() # normalize to sum (cdf)

    nclusters = 5000

    cluster_masses = np.array([np.interp(p, cdf, cluster_mass_xax) for p in np.random.rand(nclusters)])
    clusters = [make_cluster(m,mmax=m) for m in cluster_masses]

    luminosities = np.array([lum_of_cluster(c) for c in clusters])
    # no contrast
    # colors = [color_of_cluster(c) for c in clusters]

    def ctable(mass, mmin=0.08, mmax=120):
        return pl.cm.RdBu((mass-mmin)/(mmax-mmin))
        cr = np.log10(mmax)-np.log10(mmin)
        lm = np.log10(mass)-np.log10(mmin)
        return pl.cm.RdBu(lm/cr)
    colors = [color_of_cluster(c,ctable) for c in clusters]

    yax = [np.random.rand()*(np.log10(pr(m))-np.log10(pr(mmax))) + np.log10(pr(mmax)) for m in cluster_masses]

    pl.rc('font',size=30)
    pl.figure(1)
    pl.clf()
    pl.gca().set_xscale('log')

    sizes = 10**luminosities/1e5
    sizes[sizes < 10] = 10
    S = pl.scatter(cluster_masses, yax, c=colors, s=sizes, alpha=0.8)
    sm = pl.cm.ScalarMappable(cmap=pl.cm.RdBu, norm=pl.Normalize(vmin=0.08, vmax=120))
    sm._A = []
    cb = pl.colorbar(sm)
    cb.set_label("Luminosity-weighted\nMean Stellar Mass")
    pl.gca().axis([min(cluster_masses)/1.1,max(cluster_masses)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.xlabel("Cluster Mass ($M_\odot$)")
    pl.ylabel("Log(dN(M)/dM)")
    pl.savefig("plots/clusterMF_lumcolor_lumsize.png",bbox_inches='tight')

    pl.figure(2)
    pl.clf()
    pl.gca().set_xscale('log')

    sizes = cluster_masses / 50
    sizes[sizes < 10] = 10
    S = pl.scatter(cluster_masses, yax, c=colors, s=sizes, alpha=0.8)
    sm = pl.cm.ScalarMappable(cmap=pl.cm.RdBu, norm=pl.Normalize(vmin=0.08, vmax=120))
    sm._A = []
    cb = pl.colorbar(sm)
    cb.set_label("Luminosity-weighted\nMean Stellar Mass")
    pl.gca().axis([min(cluster_masses)/1.1,max(cluster_masses)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.xlabel("Cluster Mass ($M_\odot$)")
    pl.ylabel("Log(dN(M)/dM)")
    pl.savefig("plots/clusterMF_lumcolor_massize.png",bbox_inches='tight')

    pl.figure(3)
    pl.clf()
    pl.gca().set_xscale('log')

    sizes = 20*np.log(10**luminosities / cluster_masses)
    sizes[sizes < 10] = 10
    S = pl.scatter(cluster_masses, yax, c=colors, s=sizes, alpha=0.8)
    sm = pl.cm.ScalarMappable(cmap=pl.cm.RdBu, norm=pl.Normalize(vmin=0.08, vmax=120))
    sm._A = []
    cb = pl.colorbar(sm)
    cb.set_label("Luminosity-weighted\nMean Stellar Mass")
    pl.gca().axis([min(cluster_masses)/1.1,max(cluster_masses)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.xlabel("Cluster Mass ($M_\odot$)")
    pl.ylabel("Log(dN(M)/dM)")
    pl.savefig("plots/clusterMF_lumcolor_mtolsize.png",bbox_inches='tight')


    pl.show()
