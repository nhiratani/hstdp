# This is a python simulation code for the corticostriatal synapse model presented in Figure 2B.
# See the associated README file (http://github.com/nhiratani/hstdp) for the details.

from math import *
import numpy as np
import matplotlib.pyplot as plt

#PARAMETERS
dt = 0.1#time step[ms]

#decay time constants
tauc = 18.0 
taum = 3.0 
tauN = 15.0 
tauA = 3.0 
tauBP = 3.0 
tauG = 3.0 

adelay = 7.5# axonal delay
Idelay = 0.0# inhibitory delay

alphaN = 1.0# voltage dependence factor of NMDA receptor
alphaV = 2.0# voltage dependence factor of VDCC
gammaA = 1.0# AMPA coefficient
gammaN = 0.05# NMDA coefficient
gammaBP = 8.0# Backprop coefficient
gammaG = 5.0# Inhibitory heterosynaptic effect

thetap = 70# threshold for LTP
thetad = 35# threshold for LTD
Cp = 2.3# amplitude of LTP
Cd = 1.0# amplitude of LTD

#FUNCTIONS
def rk(xtmp,tautmp): #exponential decay
    kx1 = -xtmp/tautmp
    x1 = xtmp + kx1*0.5*dt

    kx2 = -x1/tautmp
    x2 = xtmp + kx2*0.5*dt

    kx3 = -x2/tautmp
    x3 = xtmp + kx3*dt

    kx4 = -x3/tautmp
    x4 = xtmp + dt*(kx1 + 2.0*kx2 + 2.0*kx3 + kx4)/6.0

    return x4

def gN(utmp): #voltage dependence of NMDA receptor
    return alphaN*utmp

def gV(utmp): #voltage dependence of VDCC
    return alphaV*utmp

def rk_cu(ctmp,utmp,xA,xN,xBP,xG): #update calcium concentration c and membrane potential u
    kc1 = -ctmp/tauc + gN(utmp)*xN + gV(utmp)
    ku1 = -utmp/taum + gammaA*xA + gammaN*gN(utmp)*xN + gammaBP*xBP - gammaG*xG
    c1 = ctmp + kc1*0.5*dt
    u1 = utmp + ku1*0.5*dt

    kc2 = -c1/tauc + gN(u1)*xN + gV(u1)
    ku2 = -u1/taum + gammaA*xA + gammaN*gN(u1)*xN + gammaBP*xBP - gammaG*xG
    c2 = ctmp + kc2*0.5*dt
    u2 = utmp + ku2*0.5*dt

    kc3 = -c2/tauc + gN(u2)*xN + gV(u2)
    ku3 = -u2/taum + gammaA*xA + gammaN*gN(u2)*xN + gammaBP*xBP - gammaG*xG
    c3 = ctmp + kc3*dt
    u3 = utmp + ku3*dt

    kc4 = -c3/tauc + gN(u3)*xN + gV(u3)
    ku4 = -u3/taum + gammaA*xA + gammaN*gN(u3)*xN + gammaBP*xBP - gammaG*xG
    c4 = ctmp + dt*(kc1 + 2.0*kc2 + 2.0*kc3 + kc4)/6.0
    u4 = utmp + dt*(ku1 + 2.0*ku2 + 2.0*ku3 + ku4)/6.0

    return c4,u4

Ts = np.arange(0,500.0,dt)

sptpre = 100.0#presynaptic spike timing
sptinh = sptpre + Idelay#inhibitory spike timing
sptposts = [80,105]#because of axonal delay, these timings correspond to [87.5, 112.5]
dts = []
for sidx in range(len(sptposts)):
    sptpost = sptposts[sidx]
    dts.append(sptpost - (sptpre-adelay))

#GRAPH SETTING
clrs = ['k','r']
    
ax1 = plt.subplot(2,2,1)
ax1.axhline(0.0,color='k')
ax1.axhline(thetad,linewidth=3.0,ls='--',color='cyan')
ax1.axhline(thetap,linewidth=3.0,ls='--',color='orange')
ax3 = plt.subplot(2,2,3)
ax3.axhline(0.0,color='k')
ax3.axhspan(-5.0,5.0,color='k',alpha=0.2)

ax2 = plt.subplot(2,2,2)
ax2.axhline(0.0,color='k')
ax2.axhline(thetad,linewidth=3.0,ls='--',color='cyan')
ax2.axhline(thetap,linewidth=3.0,ls='--',color='orange')
ax4 = plt.subplot(2,2,4)
ax4.axhline(0.0,color='k')
ax4.axhspan(-5.0,5.0,color='k',alpha=0.2)

#SIMULATION
for gidx in [0,1]:
    for sidx in range(len(sptposts)):
        sptpost = sptposts[sidx]

        Cs = []; Ys = []; dts = []
        c = 0.0 #calcium concentration
        u = 0.0 #membrane potential
        y = 0.0 #interim weight
        xA = 0.0 #inputs through AMPA receptor 
        xN = 0.0 #inputs through NMDA receptor
        xBP = 0.0 #Back-propagation
        xG = 0.0 #inhibitory heterosynaptic input
        for t in Ts:
            if abs(t - sptpre) < 0.5*dt:
                xA += 1.0; xN += 1.0
            xA = rk(xA,tauA); xN = rk(xN,tauN)
            if abs(t - sptpost) < 0.5*dt:
                xBP += 1.0
            xBP = rk(xBP,tauBP)
            if gidx == 0 and abs(t - sptinh) < 0.5*dt:
                xG += 1.0
            xG = rk(xG,tauG)

            c,u = rk_cu(c,u,xA,xN,xBP,xG)
            if int(floor(t/dt))%10 == 0:
                Cs.append(c)
                Ys.append(y)
                dts.append( t-100.0 )
            if c > thetap:
                y += Cp*dt
            if c > thetad:
                y -= Cd*dt
        
        if sidx == 0:
            ax1.plot(dts,Cs,color=clrs[gidx],linewidth=3.0)
            ax3.plot(dts,Ys,color=clrs[gidx],linewidth=3.0)
        else:
            ax2.plot(dts,Cs,color=clrs[gidx],linewidth=3.0)
            ax4.plot(dts,Ys,color=clrs[gidx],linewidth=3.0)

#GRAPH DRAWING
ax1.set_xlim(-50,300)
ax1.set_ylim(-50,150)
ax1.set_xticks([])
ax1.set_yticks([-50,0,50,100])
ax1.tick_params(labelsize=20)

ax2.set_xlim(-50,300)
ax2.set_ylim(-50,150)
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xlim(-50,300)
ax3.set_ylim(-30,30)
ax3.set_xticks([0,100,200,300])
ax3.set_yticks([-30,-15,0,15,30])
ax3.tick_params(labelsize=20)

ax4.set_xlim(-50,300)
ax4.set_ylim(-30,30)
ax4.set_xticks([0,100,200,300])
ax4.set_yticks([])
ax4.tick_params(labelsize=20)

plt.show()
