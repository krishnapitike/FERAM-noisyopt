# coding: utf-8

# In[5]:


#importing required libraries
from scipy import optimize
import numpy as np
from noisyopt import minimizeCompass
import os

#Parameters to be set
global IT, T, pNLines,pFName, binary
IT=0
binary=str("./feram ")         ##Change this to ./feram
T=10
p0=[7.99854,27.6519,-14.7591,-41.7372,0.0,0.0,94.1445,127.306,49.2177,38.4103,-137.1,-40.5909,-19.3431]
bound=0.1
nParam=len(p0)
pBounds=np.zeros((nParam,2))
lB=np.zeros(nParam)
uB=np.zeros(nParam)
for i in range(nParam):
  pBounds[i,0]=p0[i]*(1-bound)
  lB[i]       =p0[i]*(1-bound)
  pBounds[i,1]=p0[i]*(1+bound)
  uB[i]       =p0[i]*(1+bound)
pBounds=pBounds.tolist()
print(p0)
print(pBounds)
##DFT values
a0=3.880771037
Zeff=7.4996                     ##Z*
chgE=16.0217662                 ##Charge of electron in 1.0E-20 Coloumb
CbyV=Zeff*chgE/(a0**3)          ##Later to be used for polarization
Y0 = [ -0.00617745205049058, -0.00617745205049058, 0.0374533592974760, 0, 0, 0, 0, 0, 0.371610896163433 ]  ##LDA
#Y0 = [ -0.03195851288715360, -0.03195851288715360, 0.2032577423431780, 0, 0, 0, 0, 0, 0.807686208836398 ]  ##PBE
#Y0 = [ -0.01449203335146400, -0.01449203335146400, 0.0841678422141923, 0, 0, 0, 0, 0, 0.513126233218288 ]  ##PBEsol
#Y0 = [ -0.00672759523685962, -0.00672759523685962, 0.0401589740889130, 0, 0, 0, 0, 0, 0.376428804325091 ]  ##vdW-DF-C09
pNLines = 45
pInitFName = str("%04d.feram"%IT)

# In[2]:


def paramWrite(paramFName,param) :
  pFOpen = open(paramFName,"w")
  pFOpen.write("#--- Method, Temperature, and mass ---------------"+"\n")
  pFOpen.write("method = 'md'"+"\n")
  pFOpen.write("kelvin = %d"%T+"\n")
  pFOpen.write("mass_amu = 100"+"\n")
  pFOpen.write("Q_Nose = 0.1"+"\n")
  pFOpen.write("\n")
  pFOpen.write("#--- System geometry -----------------------------"+"\n")
  pFOpen.write("bulk_or_film = 'bulk'"+"\n")
  pFOpen.write("L = 16 16 16"+"\n")
  pFOpen.write("a0 =  3.880771037         latice constant a0 [Angstrom]"+"\n")
  pFOpen.write("#--- Time step -----------------------------------"+"\n")
  pFOpen.write("dt = 0.002 [pico second]"+"\n")
  pFOpen.write("n_thermalize = 60000"+"\n")
  pFOpen.write("n_average    = 20000"+"\n")
  pFOpen.write("n_coord_freq = 80000"+"\n")
  pFOpen.write("distribution_directory = 'never'"+"\n")
  pFOpen.write("\n")
  pFOpen.write("#--- On-site (Polynomial of order 8) -------------"+"\n")
  pFOpen.write("P_kappa2 = %0.16f  [eV/Angstrom2] # P_4(u) = kappa2*u2 +alpha*u4"%param[0]+"\n")
  pFOpen.write("P_alpha  = %0.16f  [eV/Angstrom4] #+gamma*(u_y*u_z+u_z*u_x+u_x*u_y)"%param[1]+"\n")
  pFOpen.write("P_gamma  = %0.16f  [eV/Angstrom4] #+gamma*(u_y*u_z+u_z*u_x+u_x*u_y)"%param[2]+"\n")
  pFOpen.write("P_k1  = %0.16f"%param[3]+"\n")
  pFOpen.write("P_k2  = %0.16f"%param[4]+"\n")
  pFOpen.write("P_k3  = %0.16f"%param[5]+"\n")
  pFOpen.write("P_k4  = %0.16f"%param[6]+"\n")
  pFOpen.write("\n")
  pFOpen.write("#--- Inter-site ----------------------------------"+"\n")
  pFOpen.write("j = -1.02221  -3.74065  0.14912  -0.40477  0.  0.10692  0."+"\n")
  pFOpen.write("\n")
  pFOpen.write("#---Elastic Constants---------------------------"+"\n")
  pFOpen.write("B11 = %0.16f"%param[7]+"\n")
  pFOpen.write("B12 = %0.16f"%param[8]+"\n")
  pFOpen.write("B44 = %0.16f  [eV]"%p0[9]+"\n")
  pFOpen.write("\n")
  pFOpen.write("#--- Elastic Coupling ----------------------------"+"\n")
  pFOpen.write("B1xx = %0.16f  [eV/Angstrom2]"%param[10]+"\n")
  pFOpen.write("B1yy = %0.16f  [eV/Angstrom2]"%param[11]+"\n")
  pFOpen.write("B4yz = %0.16f  [eV/Angstrom2]"%param[12]+"\n")
  pFOpen.write("\n")
  pFOpen.write("#--- Dipole --------------------------------------"+"\n")
  pFOpen.write("seed = 1242914819 1957271599"+"\n")
  pFOpen.write("init_dipo_avg = 0.001   0.001   0.333    [Angstrom]  # Average   of initial dipole displacements"+"\n")
  pFOpen.write("init_dipo_dev = 0.0002  0.0002  0.002   [Angstrom]  # Deviation of initial dipole displacements"+"\n")
  pFOpen.write("Z_star        = 7.4996"+"\n")
  pFOpen.write("epsilon_inf   = 8.551"+"\n")
  pFOpen.close()
  print("Writing",paramFName,"Done")


##Loops until avg file is found and reads strains and dipole moments from the avg file
def findAvg() :
  global IT
  Y = np.zeros(9)
  while 1 :
    avgFName=str("%04d.avg"%IT)
    #print(avgFName)
    runCommand2=str('find ./ -name '+avgFName)
    if len( os.popen(runCommand2).read() ) :
      ##script to read avg file
      avgFOpen=open(avgFName,'r')              ##opening  avg file
      for line in avgFOpen :                   ##reading line by line
        line=line.split()                    ##only one line to read in avg file
        for i in range(9) :                  ##6 strain components to read
          Y[i] = float( line[ i+4 ] ) ##5th item in line is first component etc.
      break
  return (Y)

def readLog() :
  ## for now nothing to do here
  ## if required energy can be read from here
  return

# In[10]:

def Fun(param) :
  global IT                     ##IT is incremented before return
  pFName=str("%04d.feram"%IT)
  print(pFName)
  paramWrite(pFName,param)
  runCommand1=str(binary+pFName)
  os.popen(runCommand1)
  Y=findAvg()
  print(Y)
  #readLog()                     ##We are doing nothing here for now
  ##converting polar displacemt to polarization for computing error
  ##Not required for now ; comparing the polar displacements with DFT polar displacements
  #Y[6]=Y[6]*CbyV/((Y[0]+1)*(Y[1]+1)*(Y[2]+1))
  #Y[7]=Y[7]*CbyV/((Y[0]+1)*(Y[1]+1)*(Y[2]+1))
  #Y[8]=Y[8]*CbyV/((Y[0]+1)*(Y[1]+1)*(Y[2]+1))
  ##script to compute error
  totalError = 0
  for i in [0,1,2,8] :
    totalError = totalError + ( ( Y[i] - Y0[i] ) / ( Y0[i] ) )**2
  IT=IT+1
  return(totalError)

#paramWrite(pInitFName,p0)       ##note required, defined n the function itself
#fopt.Bounds(lB, uB, keep_feasible=False)
#fopt = optimize.minimize(Fun,p0,method='L-BFGS-B')
fopt = minimizeCompass(Fun,bounds=pBounds, x0=p0, deltatol=0.1, paired=False)
#fopt.Bounds(lB, uB, keep_feasible=False)

#printing
print("###################################################################")
print("Minimum function value      =", fopt.fun)
print("Minimum function value at x =", fopt.x)
print("Number of iterations done   =", IT)
#print(xopt)
print("###################################################################")


# In[ ]:
