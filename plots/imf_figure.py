from agpy.imf import coolplot,kroupa,make_cluster


if __name__ == "__main__":
    from pylab import *
    rc('font',size=30)
    figure(1)
    clf()
    cluster,yax,colors = coolplot(1000, massfunc=kroupa)
    scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85)
    gca().set_xscale('log')

    masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

    plot(masses,np.log10(kroupa(masses)),'r--',linewidth=2,alpha=0.5)
    xlabel("Stellar Mass")
    ylabel("log(dN(M)/dM)")
    gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])

    #figure(2)
    #clf()
    #def cloud_massfunc(mass,m0=1e3,alpha=1.1):
    #    return (mass/m0)**-alpha

    #clouds = make_cluster(1e8, massfunc=cloud_massfunc)
