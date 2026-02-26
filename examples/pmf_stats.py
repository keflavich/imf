import numpy as np

import imf.imf,imf.pmf
from imf import chabrier2005,Kroupa
from imf.pmf import PMF,PMF_2C

def stats(mf,desc,**kwargs):
    total = mf.m_integrate(mmin, mmax, **kwargs)[0]
    gt10 = mf.m_integrate(10, mmax, **kwargs)[0]
    print(f'Mass fraction M>10 ({desc}) = {gt10/total}')

def accel_stats(mf,desc):
    for tau in taus:
        mf.tau = tau
        stats(mf,f'{desc}_accel, tau = {tau}',accelerating=True)

mmin = 0.033
mmax = 120

taus = [0.1,0.5,1,2]

chabrier = chabrier2005
chabrier.mmin = mmin
chabrier.mmax = mmax
chabrier.normalize()

kroupa = Kroupa(mmin=mmin,mmax=mmax)
kroupa.normalize()

stats(chabrier,'chabrier')
pmf = PMF(chabrier,history='is')
stats(pmf,'chabrier_is')
accel_stats(pmf,'chabrier_is')
pmf.history = 'tc'
stats(pmf,'chabrier_tc')
accel_stats(pmf,'chabrier_tc')
pmf.history = 'ca'
stats(pmf,'chabrier_ca')
accel_stats(pmf,'chabrier_ca')
pmf = PMF_2C(chabrier,history='tc')
stats(pmf,'chabrier_2ctc')
accel_stats(pmf,'chabrier_2ctc')

stats(kroupa,'kroupa')
pmf = PMF(kroupa,history='is')
stats(pmf,'kroupa_is')
accel_stats(pmf,'kroupa_is')
pmf.history = 'tc'
stats(pmf,'kroupa_tc')
accel_stats(pmf,'kroupa_tc')
pmf.history = 'ca'
stats(pmf,'kroupa_ca')
accel_stats(pmf,'kroupa_ca')
pmf = PMF_2C(kroupa,history='tc')
stats(pmf,'kroupa_2ctc')
accel_stats(pmf,'kroupa_2ctc')
