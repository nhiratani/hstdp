# This is a python simulation code for the schaffercollateral synapse model presented in Figure 2D.
# See the associated README file(http://github.com/nhiratani/hstdp) for the details.

from math import *
import numpy as np
import matplotlib.pyplot as plt

#PARAMETERS
dt = 0.1 #time step[ms]
N = 2

#decay time constants
tauc = 18.0
taum = 3.0
tauN = 15.0
tauA = 3.0
tauBP = 3.0
tauG = 3.0
tauE = 6.0

adelay = 7.5# axonal delay
postdelay = 10.0# pre-post delay
Edelay = 1.0# inhibitory delay

alphaN = 1.0# voltage dependence factor of NMDA receptor
alphaV = 2.0# voltage dependence factor of VDCC
gammaA = 1.0# AMPA coefficient
gammaN = 0.20# NMDA coefficient
gammaBP = 8.5# Backprop coefficient
gammaG = 3.0# Inhibitory heterosynaptic effect
gammaE = 1.0# Excitatory heterosynaptic effect

thetap = 70# threshold for LTP
thetad = 35# threshold for LTD
Cp = 2.2# amplitude of LTP
Cd = 1.0# amplitude of LTD

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

def rk_cu(ctmp,utmp,xA,xN,xBP,xG,xE): #update calcium concentration c and membrane potential u
    kc1 = -ctmp/tauc + gN(utmp)*xN + gV(utmp)
    ku1 = -utmp/taum + gammaA*xA + gammaN*gN(utmp)*xN + gammaBP*xBP - gammaG*xG + gammaE*xE
    c1 = ctmp + kc1*0.5*dt
    u1 = utmp + ku1*0.5*dt

    kc2 = -c1/tauc + gN(u1)*xN + gV(u1)
    ku2 = -u1/taum + gammaA*xA + gammaN*gN(u1)*xN + gammaBP*xBP - gammaG*xG + gammaE*xE
    c2 = ctmp + kc2*0.5*dt
    u2 = utmp + ku2*0.5*dt

    kc3 = -c2/tauc + gN(u2)*xN + gV(u2)
    ku3 = -u2/taum + gammaA*xA + gammaN*gN(u2)*xN + gammaBP*xBP - gammaG*xG + gammaE*xE
    c3 = ctmp + kc3*dt
    u3 = utmp + ku3*dt

    kc4 = -c3/tauc + gN(u3)*xN + gV(u3)
    ku4 = -u3/taum + gammaA*xA + gammaN*gN(u3)*xN + gammaBP*xBP - gammaG*xG + gammaE*xE
    c4 = ctmp + dt*(kc1 + 2.0*kc2 + 2.0*kc3 + kc4)/6.0
    u4 = utmp + dt*(ku1 + 2.0*ku2 + 2.0*ku3 + ku4)/6.0

    return c4,u4

Ts = np.arange(0,500.0,dt)

sptpres = []
for i in range(N):
    sptpres.append(-500.0)
    if i == 0:
        sptpres[i] = 100.0 + adelay
sptposts = [90,105]
dts = []
for sidx in range(len(sptposts)):
    sptpost = sptposts[sidx]
    dts.append(sptpost - (sptpres[0]-adelay))

#GRAPH SETTING
clrs = ['b','g']

ax1 = plt.subplot(2,2,1)
ax1.axhline(0.0,color='k')
ax1.axhline(thetad,linewidth=3.0,ls='--',color='cyan')
ax1.axhline(thetap,linewidth=3.0,ls='--',color='orange')
ax3 = plt.subplot(2,2,3)
ax3.axhline(0.0,color='k')
ax3.axhspan(-15.0,15.0,color='k',alpha=0.2)

ax2 = plt.subplot(2,2,2)
ax2.axhline(0.0,color='k')
ax2.axhline(thetad,linewidth=3.0,ls='--',color='cyan')
ax2.axhline(thetap,linewidth=3.0,ls='--',color='orange')
ax4 = plt.subplot(2,2,4)
ax4.axhline(0.0,color='k')
ax4.axhspan(-15.0,15.0,color='k',alpha=0.2)

#SIMULATION
for gidx in [0,1]:
    for sidx in range(len(sptposts)):
        sptpost = sptposts[sidx]
        sptinh = sptpost - postdelay

        ts = []; Cs = []; Ys = []
        for i in range(N):
            Cs.append([]); Ys.append([])
        c = [] #calcium concentration
        u = [] #membrane potential
        y = [] #interim weight
        xA = [] #inputs through AMPA receptor 
        xN = [] #inputs through NMDA receptor
        xBP = [] #Back-propagation
        xG = [] #inhibitory heterosynaptic input
        xE = [] #excitatory heterosynaptic input
        for i in range(N):
            c.append(0.0); u.append(0.0); y.append(0.0)
            xA.append(0.0); xN.append(0.0); xBP.append(0.0)
            xG.append(0.0); xE.append(0.0)
        
        for t in Ts:
            for i in range(N):
                if abs(t - sptpres[i]) < 0.5*dt:
                    xA[i] += 1.0; xN[i] += 1.0
                xA[i] = rk(xA[i],tauA); xN[i] = rk(xN[i],tauN)
                if abs(t - sptpost) < 0.5*dt:
                    xBP[i] += 1.0
                xBP[i] = rk(xBP[i],tauBP)

                if gidx == 0:
                    if abs(t - sptinh) < 0.5*dt:
                        xG[i] += 1.0
                    xG[i] = rk(xG[i],tauG)
                else:
                    xG[i] = 0.0

                for j in range(N):
                    if i != j:
                        if abs(t - sptpres[j] - Edelay) < 0.5*dt:
                            xE[i] += 1.0
                xE[i] = rk(xE[i],tauE)

                c[i],u[i] = rk_cu(c[i],u[i],xA[i],xN[i],xBP[i],xG[i],xE[i])
                if int(floor(t/dt))%10 == 0:
                    Cs[i].append(c[i])
                    Ys[i].append(y[i])
                    if i == 0:
                        ts.append( t - 100.0 )
                if c[i] > thetap:
                    y[i] += Cp*dt
                if c[i] > thetad:
                    y[i] -= Cd*dt

        if gidx == 0:
            if sidx == 0:
                for i in range(N):
                    ax1.plot(ts,Cs[i],color=clrs[i],linewidth=3.0)
                    ax3.plot(ts,Ys[i],color=clrs[i],linewidth=3.0)
            else:
                for i in range(N):
                    ax2.plot(ts,Cs[i],color=clrs[i],linewidth=3.0)
                    ax4.plot(ts,Ys[i],color=clrs[i],linewidth=3.0)
        else:
            if sidx == 0:
                for i in range(N):
                    ax1.plot(ts,Cs[i],color=clrs[i],ls='--',linewidth=3.0)
                    ax3.plot(ts,Ys[i],color=clrs[i],ls='--',linewidth=3.0)
            else:
                for i in range(N):
                    ax2.plot(ts,Cs[i],color=clrs[i],ls='--',linewidth=3.0)
                    ax4.plot(ts,Ys[i],color=clrs[i],ls='--',linewidth=3.0)

#GRAPH DRAWING
ax1.set_xlim(-50,200)
ax1.set_ylim(-40,220)
ax1.set_xticks([])
ax1.set_yticks([0,50,100,150,200])
ax1.tick_params(labelsize=20)

ax2.set_xlim(-50,200)
ax2.set_ylim(-40,220)
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xlim(-50,200)
ax3.set_ylim(-40,40)
ax3.set_xticks([0,100,200])
ax3.set_yticks([-40,-20,0,20,40])
ax3.tick_params(labelsize=20)

ax4.set_xlim(-50,200)
ax4.set_ylim(-40,40)
ax4.set_xticks([0,100,200])
ax4.set_yticks([])
ax4.tick_params(labelsize=20)
plt.show()
