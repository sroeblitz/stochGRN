#Strasser et al. (2012). Stability and Multiattractor Dynamics of a Toggle Switch
# Based on a Two-Stage Model of Stochastic Gene Expression
# doi: 10.1016/j.bpj.2011.11.4000

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import random
import math

np.random.seed(10)

rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 18


alpha=0.05		#sec^-1
beta=0.05		#sec^-1 mRNA^-1
gamma=0.005		#sec^-1
delta=0.0005		#sec^-1
taup=1			#sec^-1 protein^-1
taum=0.1		#sec^-1


def gillespie_toggle_switch(x0,tf):

    #stoichiometric vectors
    Nr=np.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,-1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,0,0,-1],[0,-1,0,0,-1,0],[-1,0,0,0,0,-1],[0,1,0,0,1,0],[1,0,0,0,0,1]])
         
    DA=x0[0]
    DB=x0[1]
    MA=x0[2]
    MB=x0[3]
    PA=x0[4]
    PB=x0[5]
    x=x0			#current state
    X=[x0]			#array with all states visited
    t=0				#current time
    tvec=[] 			#array of reaction time events	
    tvec.append(t)    

    
    i=1
    while (t<tf):
        i=i+1
        #propensities
        propensities=np.array([alpha*DA,alpha*DB,gamma*MA,gamma*MB,beta*MA,beta*MB,delta*PA,delta*PB,taup*DB*PA,taup*DA*PB,taum*(1-DB),taum*(1-DA)])
        for k in range (0,12,1):
            #exclude unfeasible reactions that would result in negative copy numbers 
            if (np.amin(x+Nr[k])<0): 
                propensities[k]=0
        #total reaction intensity
        W=np.sum(propensities)
        if (W==0):
           #warning('negative copy number')
           break 
        if (W>0):
            tau=-np.log(np.random.uniform(0,1,1))/W  #when does the next reaction take place?
            Wrand=W*np.random.uniform(0,1,1)
            r=np.searchsorted(np.cumsum(propensities),Wrand)
            x=x+Nr[r]    	#update state vector
            t=t+tau		#update time
            DA=x[0,0]		#update species for calculation of propensities
            DB=x[0,1]
            MA=x[0,2]
            MB=x[0,3]
            PA=x[0,4]
            PB=x[0,5]
            x=[DA,DB,MA,MB,PA,PB]
            X=np.append(X,[x],axis=0)	#append new state to aray of all visisted states
            tvec.append(t)		#append current time to list of all reaction times
    return X,tvec

#initual values
DA=0
DB=0
PA=100
PB=100	#0.01
MA=0.01*PA
MB=0.01*PB

x0=[DA,DB,MA,MB,PA,PB]

#final simulation time
tf=10*60*60	#10 hours

numTraj=25			#number of trajectories to be generated 
PA_list=[]
PB_list=[]
time_list=[]

#store solution every tstep time units to save memory
tstep=60 	#1 hour

#run multiple trajectories
for traj in range (0,numTraj,1):   
    print('traj:',traj)
    PA_list.append(x0[4])
    PB_list.append(x0[5])
    time_list.append(traj*tf)
    #perform SSA until final time tf
    xx,tt = gillespie_toggle_switch(x0,tf)
    x0=xx[-1,:]
    for t in range (tstep,tf,tstep):
        ind=np.max(np.argwhere(np.asarray(tt) < t))
        PA_list.append(xx[ind,4])
        PB_list.append(xx[ind,5])
        time_list.append(traj*tf+tt[ind])
    
    

PA_values=np.asarray(PA_list)
PB_values=np.asarray(PB_list)
time_values=np.asarray(time_list)


def model(t,y):
    DA=y[0]
    DB=y[1]
    MA=y[2]
    MB=y[3]
    PA=y[4]
    PB=y[5]
    DAdt=taum*(1-DA)-taup*DA*PB
    DBdt=taum*(1-DB)-taup*DB*PA
    MAdt=alpha*DA-gamma*MA
    MBdt=alpha*DB-gamma*MB
    PAdt=beta*MA-delta*PA+taum*(1-DB)-taup*DB*PA
    PBdt=beta*MB-delta*PB+taum*(1-DA)-taup*DA*PB
    return [DAdt,DBdt,MAdt,MBdt,PAdt,PBdt]


# time span
tspan=[0,tf]

# solve ODE
sol=solve_ivp(model,tspan,x0,method='Radau',rtol=1e-6,atol=1e-6)


graph=plt.figure(1)
#plt.plot(sol.t,sol.y[4,:]-sol.y[5,:],'r-')
plt.plot(time_values/(60*60),PA_values-PB_values)
plt.ylabel('PA(t)-PB(t)')
plt.xlabel('time in hours')
plt.show()
#graph.savefig('toggle_switch_timecourse.png')


#histogram of final states
hist=plt.figure(2)
plt.hist(PA_values-PB_values,bins=20,density=True,label='SSA histogram')
plt.xlabel('PA-PB')
plt.ylabel('relative frequency')
plt.show()
#hist.savefig('toggle_switch_hist.png')

'''
#some phase plane plots:

#compute steady state
eta=math.sqrt((4*alpha*beta*taup)/(gamma*delta*taum)+1)
ss=-taum/(2*taup)*(1-eta)
print('steady state: ',ss)

graph=plt.figure(3)
plt.yscale("log")
plt.xscale("log")
plt.plot(sol.y[4,:],sol.y[5,:],'r-')
#plt.plot(tt,xx[:,0])
plt.plot(ss,ss,'ko')
plt.ylabel('PB')
plt.xlabel('PA')
plt.show()

P0=np.array([[100,1],[10,1],[1,1],[1,10],[1,100],[10,100],[100,100],[100,10],[100,1]])
graph=plt.figure(4)
plt.yscale("log")
plt.xscale("log")
for k in range (0,len(P0),1):
    print(k)
    P=P0[k]
    #print(P)
    x0[4]=P[0]
    x0[5]=P[1]
    sol=solve_ivp(model,tspan,x0,method='BDF',rtol=1e-6,atol=1e-6)
    plt.plot(sol.y[4,:],sol.y[5,:])
plt.plot(ss,ss,'ko')
plt.ylabel('PB')
plt.xlabel('PA')
plt.show()
'''

