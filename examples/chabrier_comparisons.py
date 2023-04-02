"""
Compare the Chabrier distribution pulled from eqn 18 of Chabrier 2003 to
that quoted in McKee & Offner 2010 as "Chabrier 2005"
"""
import imf
import numpy as np
import pylab as pl

chabrier = imf.chabrierpowerlaw
# Eq2 of McKee & Offner use psi(m) d ln m, but eqn 51 - which provides these numbers - gives psi as a function of log10
chabrier2005 = imf.ChabrierPowerLaw(lognormal_width=0.55*np.log(10),
                                    lognormal_center=0.2,
                                    alpha=2.35)
kroupa_lognormal = imf.ChabrierPowerLaw(lognormal_width=(0.75)*np.log(10),
                                        lognormal_center=0.055*np.log(10),
                                        alpha=2.3)
chabrier_single = imf.ChabrierPowerLaw(lognormal_width=(0.69)*np.log(10),
                                       lognormal_center=0.079*np.log(10),
                                       alpha=2.3)

masses = np.geomspace(0.01, 10, 1000)
pl.figure(1).clf()
pl.loglog(masses, chabrier(masses), label='Chabrier 2003 eqn 18')
pl.loglog(masses, chabrier2005(masses), label='Chabrier 2005 via McKee & Offner 2010')
pl.xlabel("Mass")
pl.ylabel("$\\xi \\equiv dN/dM$")
pl.legend(loc='best')
pl.savefig("Chabrier2003v2005.png")

pl.figure(2).clf()
pl.loglog(masses, chabrier2005(masses) - chabrier(masses), label='C03-C05')
pl.loglog(masses, chabrier(masses) - chabrier2005(masses), label='C05-C03')
pl.xlabel("Mass")
pl.ylabel("$\\xi_1 - \\xi_2$")
pl.legend(loc='best')
pl.savefig("Chabrier2003v2005_diff.png")

pl.figure(3).clf()
pl.loglog(masses, (chabrier2005(masses) - chabrier(masses))/chabrier(masses), label='(C03-C05)/C03')
pl.loglog(masses, (chabrier(masses) - chabrier2005(masses))/chabrier(masses), label='(C05-C03)/C03')
pl.xlabel("Mass")
pl.ylabel("$(\\xi_1-\\xi_2)/\\xi_1$")
pl.legend(loc='best')
pl.savefig("Chabrier2003v2005_relativediff.png")

pl.figure(4).clf()
pl.loglog(masses, chabrier(masses), label='Chabrier 2003 eqn 18')
pl.loglog(masses, chabrier2005(masses), label='Chabrier 2005 via McKee & Offner 2010')
pl.loglog(masses, kroupa_lognormal(masses), label='Kroupa lognormal')
pl.loglog(masses, chabrier_single(masses), label='Chabrier single')
pl.loglog(masses, imf.kroupa(masses), label='Kroupa canonical')
pl.xlabel("Mass")
pl.ylabel("$\\xi \\equiv dN/dM$")
pl.legend(loc='best')
pl.savefig("Chabrier2003v2005vsKroupa.png")

pl.figure(5).clf()
pl.loglog(masses, masses*chabrier(masses), label='Chabrier 2003 eqn 18')
pl.loglog(masses, masses*chabrier2005(masses), label='Chabrier 2005 via McKee & Offner 2010')
pl.loglog(masses, masses*kroupa_lognormal(masses), label='Kroupa lognormal')
pl.loglog(masses, masses*chabrier_single(masses), label='Chabrier single')
pl.loglog(masses, masses*imf.kroupa(masses), label='Kroupa canonical')
pl.xlabel("Mass")
pl.ylabel("$M \\xi \\equiv dN/dM$")
pl.legend(loc='best')
pl.savefig("Chabrier2003v2005vsKroupa.png")

