#Ozbudak et al. (2002). Regulation of noise in the expression of a single gene
#DOI: 10.1038/ng869
#simulate bursts of protein creation of average size b=kP/gammaR occurring at average rate kR

import numpy as np
from scipy.integrate import solve_ivp
#from array import array
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import random

np.random.seed(10)

rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 18

gammaR=0.1 #s^-1, mRNA degradation rate
gammaP=0.002 #s^-1

#low transcription but high translation rates
kR=0.01 #s^-1
b=10

#high transcription but low translation rates
#kR=0.1 #s^-1
#b=1

kP=b*gammaR


def gillespie_gene_expression(x0,tf):

    #stoichiometric vectors
    Nr=np.array([[1,0],[-1,0],[0,1],[0,-1]])
         
    R=x0[0]			#RNA
    P=x0[1]			#protein
    x=x0			#current state
    X=[x0]			#array with all states visited
    t=0				#current time
    tvec=np.empty([1,1])	#array of reaction time events	
    tvec[0]=t
    #tvec=[t]
    
    i=1
    while (t<tf):
        i=i+1
        #propensities
        alpha=np.array([kR,gammaR*R,kP*R,gammaP*P])
        for k in range (0,4,1):
            #exclude unfeasible reactions that would result in negative copy numbers 
            if (np.amin(x+Nr[k])<0): 
                alpha[k]=0
        #total reaction intensity
        W=np.sum(alpha)
        if (W==0):
           #warning('negative copy number')
           break 
        if (W>0):
            tau=-np.log(np.random.uniform(0,1,1))/W  #when does the next reaction take place?
            Wrand=W*np.random.uniform(0,1,1)
            r=np.searchsorted(np.cumsum(alpha),Wrand) #which reaction fires?
            x=x+Nr[r]    	#update state vector
            t=t+tau		#update time
            R=x[0,0]		#update R for calculation of propensities
            P=x[0,1]		#update P for calculation of propensities
            x=[R,P]
            X=np.append(X,[x],axis=0)	#append new state to aray of all visisted states
            tvec=np.append(tvec,[t],axis=0)	#append current time to array of all reaction times
    return X,tvec

#initual values
R=1	#RNA
P=50	#protein
x0=[R,P]

#final simulation time
tf=8*60*60 #8 hours

numTraj=1			#number of trajectories to be generated 

for traj in range (0,numTraj,1):   #run multiple trajectories
    print('traj:',traj)
    #perform SSA until final time tf
    xx,tt=gillespie_gene_expression(x0,tf)
    #plt.semilogy(tt,xx[:,1])
    


#ODE model
def model(t,y):
    dRdt = kR-gammaR*y[0]
    dPdt = kP*y[0]-gammaP*y[1]
    return [dRdt,dPdt]

# initial condition
y0 = [R,P]

# time span
tspan=[0,tf]

# solve ODE
sol=solve_ivp(model,tspan,y0,method='LSODA',rtol=1e-6,atol=1e-6)


graph=plt.figure(1)
plt.plot(sol.t,sol.y[1,:],'r-')
plt.plot(tt,xx[:,1])
plt.ylim(0,160)
plt.ylabel('protein number')
plt.xlabel('time in sec')
plt.show()
#graph.savefig('stochastic-gene-expression2.png')

#sample values equidistant in time to get right histogram
dt=1
N=int(np.floor(tf/dt)+1)
x_values=np.empty(N)
for k in range(0,tf,dt):
    idx=np.searchsorted(tt[:,0],k*dt)
    x_values[k]=xx[idx,1]

#histogram of final states
hist=plt.figure(2)
#plt.hist(xx[:,1],bins=20,density=True,label='SSA histogram')
plt.hist(x_values,bins=20,density=True,label='SSA histogram')
plt.xlim(0,160)
plt.xlabel('protein number')
plt.ylabel('relative frequency')
plt.show()
#hist.savefig('stochastic-gene-expression-hist2.png')




