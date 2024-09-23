#selv-activating gene
#Hermsen et al. (2011). Speed, Sensitivity, and Bistability in Auto-activating Signaling Circuits.
#doi:10.1371/journal.pcbi.1002265

import numpy as np
from scipy.integrate import solve_ivp
#from array import array
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

random.seed(10)

rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 18

r=0.45		#fraction of activated transcription factors (TF)
b=10		#each mRNA transcribed from the promoter is instantly translated b times (the ‘‘burst size’’)
V=10		#volume of the cell
alpha=1		#min^-1, maximal transcription rate at full activation
beta=0.04	#min^-1, degradation rate constant of the TF
K=50/V		#dissociation constant of the modified TF binding to its operator
f=50		#maximal fold change of the promoter, >alpha/beta for bimodality
H=2		#Hill coefficient

'''
alpha=4
K=200
V=1
f=600
r=0.25
'''

def gillespie_gene_expression(x0,tf):

    #stoichiometric vectors
    Nr=np.array([b,-1])
         
    n=x0[0]
    X=np.empty([1,1])
    X[0]=x0			#array with all states visited
    t=0				#current time
    tvec=np.empty([1,1])	#array of reaction time events	
    tvec[0]=t
        
    i=1
    while (t<tf):
        i=i+1
        #propensities
        g=alpha*((r*n/V/K)**H+1/f)/((r*n/V/K)**H+1)
        propensities=np.array([g,beta*n])
        for k in range (0,2,1):
            #exclude unfeasible reactions that would result in negative copy numbers 
            if (np.amin(n+Nr[k])<0): 
                propensities[k]=0
        #total reaction intensity
        W=np.sum(propensities)
        if (W==0):
           #warning('negative copy number')
           break 
        if (W>0):
            tau=-np.log(np.random.uniform(0,1,1))/W  #when does the next reaction take place?
            Wrand=W*np.random.uniform(0,1,1)
            idx=np.searchsorted(np.cumsum(propensities),Wrand)	#which reaction fires?
            n=n+Nr[idx]    	#update state vector
            t=t+tau		#update time
            X=np.append(X,[n],axis=0)	#append new state to aray of all visisted states
            tvec=np.append(tvec,[t],axis=0)	#append current time to array of all reaction times
    return X,tvec

#initual values
y0=[1]

#final simulation time in minutes
tf=24*60*10	#10 days

numTraj=1			#number of trajectories to be generated 

for traj in range (0,numTraj,1):   #run multiple trajectories
    print('traj:',traj)
    #perform SSA until final time tf
    xx,tt=gillespie_gene_expression(y0,tf)
    #plt.semilogy(tt,xx[:,1])
    


#ODE model
def ODEmodel(t,y):
    dydt = alpha*((r*y/K)**H+1/f)/((r*y/K)**H+1)*b/V-beta*y
    return dydt


# time span
tspan=[0,tf]

# solve ODE
sol=solve_ivp(ODEmodel,tspan,y0=[20],method='LSODA',rtol=1e-6,atol=1e-6)
#sol2=solve_ivp(ODEmodel,tspan,y0=[50],method='LSODA',rtol=1e-6,atol=1e-6)

graph=plt.figure(1)
plt.plot(sol.t,sol.y[0,:],'r-')
#plt.plot(sol2.t,sol2.y[0,:],'b-')
plt.plot(tt,xx[:,0])
plt.ylabel('TF')
plt.xlabel('time in min')
plt.show()
#graph.savefig('stochastic-gene-expression2.png')


#sample values equidistant in time to get right histogram
dt=1
N=int(np.floor(tf/dt)+1)
x_values=np.empty(N)
for k in range(0,tf,dt):
    idx=np.searchsorted(tt[:,0],k*dt)
    x_values[k]=xx[idx,0]
    
    

#histogram of states
hist=plt.figure(2)
#plt.hist(xx[:,0],bins=20,density=True,label='SSA histogram')
plt.hist(x_values,bins=20,density=True,label='SSA histogram')
#plt.xlim(0,160)
plt.xlabel('protein number')
plt.ylabel('relative frequency')
plt.show()
#hist.savefig('stochastic-gene-expression-hist2.png')




