#Raj et al (2006). Stochastic mRNA Synthesis in Mammalian Cells
#DOI: 10.1371/journal.pbio.0040309 


import numpy as np
from scipy.integrate import solve_ivp
#from array import array
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import random

np.random.seed(10)

rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 18

#parameter values as described in the paper
delta=0.01		#mRNA degradation rate
la=2.44*delta		#gene activation rate
gamma=2.49*delta	#gene inactivation rate
mu=910*delta		#mRNA sysnthesis rate
#muP=100*delta		#protein synthesis rate
#deltaP=0.02*delta	#prodein degradation rate
f=1			#f=1 no auto-activation
muP=2*delta
deltaP=muP
K=50
n=2

#own parameter values
mu=10
delta=0.01
la=0.25*delta	#0.02
gamma=0.25*delta	#0.05
muP=delta
deltaP=0.1*delta


def gillespie_2state_gene_expression(x0,tf):

    #stoichiometric vectors  for state (I,A,M,P)
    #Nr=np.array([[-1,1,0,0],[1,-1,0,0],[0,0,1,0],[0,0,-1,0]])	#without protein dynamics
    Nr=np.array([[-1,1,0,0],[1,-1,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]])

    I=x0[0]			#inactive gene
    A=x0[1]			#active gene
    M=x0[2]			#mRNA	
    P=x0[3]			#protein
    x=x0			#current state
    X=[x0]			#array with all states visited
    t=0				#current time
    tvec=np.empty([1,1])	#array of reaction time events	
    tvec[0]=t
   
    
    i=1
    while (t<tf):
        i=i+1
        #propensities
        g=((P/K)**n+1/f)/((P/K)**n+1)
        alpha=np.array([la*g*I,gamma*A,mu*A,delta*M,muP*M,deltaP*P])
        #alpha=np.array([la*g*I,gamma*A,mu*A,delta*M])
        for k in range (0,len(alpha),1):
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
            r=np.searchsorted(np.cumsum(alpha),Wrand)	#which reaction fires next
            x=x+Nr[r]    	#update state vector
            t=t+tau		#update time
            I=x[0,0]		#update I for calculation of propensities
            A=x[0,1]		#update A for calculation of propensities
            M=x[0,2]		#update I for calculation of propensities
            P=x[0,3]		#update A for calculation of propensities
            x=[I,A,M,P]
            X=np.append(X,[x],axis=0)	#append new state to aray of all visisted states
            tvec=np.append(tvec,[t],axis=0)	#append current time to array of all reaction times
    return X,tvec

#initual values
I=1
A=0
M=0
P=0
x0=[I,A,M,P]

#final simulation time in sec
tf=60*60

numTraj=1			#number of trajectories to be generated 

for traj in range (0,numTraj,1):   #run multiple trajectories
    print('traj:',traj)
    #perform SSA until final time tf
    xx,tt=gillespie_2state_gene_expression(x0,tf)
    #plt.semilogy(tt,xx[:,1])
    


#ODE model
def model(t,y):
    dIdt = -la*y[0] + gamma*y[1]
    dAdt = la*y[0] - gamma*y[1]
    dMdt = mu*y[1] - delta*y[2]
    dPdt = muP*y[2] - deltaP*y[3]
    return [dIdt,dAdt,dMdt,dPdt]

# initial condition
y0 = [I,A,M,P]

# time span
tspan=[0,tf]

# solve ODE
sol=solve_ivp(model,tspan,y0,method='LSODA',rtol=1e-6,atol=1e-6)

graph=plt.figure(3)
#plt.plot(sol.t,sol.y[2,:],'r-')
plt.plot(tt,xx[:,2])
plt.ylabel('mRNA number')
plt.xlabel('time in sec')
plt.show()

graph=plt.figure(1)
#plt.plot(sol.t,sol.y[3,:],'r-')
plt.plot(tt,xx[:,3])
plt.ylabel('protein number')
plt.xlabel('time in sec')
plt.show()
#graph.savefig('stochastic-2state-gene-expression2.png')


#sample values equidistant in time to get right histogram
dt=1
N=int(np.floor(tf/dt)+1)
x_values=np.empty(N)
for k in range(0,tf,dt):
    idx=np.searchsorted(tt[:,0],k*dt)
    x_values[k]=xx[idx,2]	#column 2 for mRNA
    

#histogram of final states
hist=plt.figure(2)
#plt.hist(xx[:,2],bins=20,density=True,label='SSA histogram')
plt.hist(x_values,bins=20,density=True,label='SSA histogram')
#plt.xlabel('protein number')
plt.xlabel('mRNA number')
plt.ylabel('relative frequency')
plt.show()
#hist.savefig('stochastic-2state-gene-expression-hist2.png')




