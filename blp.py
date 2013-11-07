import os
import numpy as np
import scipy.io as io
import pandas as pd
from scipy import optimize

class BLP():
    def __init__(self):
        ### Variable loadings and definitions. Much of the text is borrowed from
        ### http://emlab.berkeley.edu/users/bhhall/e220c/readme.html
        ps2 = io.loadmat('ps2.mat')
        iv = io.loadmat('iv.mat')
        
        # x1 variables enter the linear part of the estimation and include price
        # variable and 24 brand dummies. 
        self.x1 = ps2['x1']
        
        # x2 variables enter the non-linear part and include a constant, price,
        # sugar content, a mushiness dummy. 
        self.x2 = ps2['x2']
        
        # the market share of brand j in market t
        self.s_jt = ps2['s_jt']

        # Random draws. For each market 80 iid normal draws are provided.
        # They correspond to 20 "individuals", where for each individual
        # there is a different draw for each column of x2. 
        v = ps2['v']

        # draws of demographic variables from the CPS for 20 individuals in each
        # market. The first 20 columns give the income, the next 20 columns the
        # income squared, columns 41 through 60 are age and 61 through 80 are a
        # child dummy variable. 
        demogr = ps2['demogr']

        self.vfull = np.array([np.repeat(i,24).reshape(80,24) for i in v]).swapaxes(1,2).reshape(2256,80)
        self.dfull = np.array([np.repeat(i,24).reshape(80,24) for i in demogr]).swapaxes(1,2).reshape(2256,80)
        
        self.IV = np.matrix(np.concatenate((iv['iv'][:,1:],self.x1[:,1:].todense()),1))

        self.ns = 20       # number of simulated "indviduals" per market 
        self.nmkt = 94     # number of markets = (# of cities)*(# of quarters)  
        self.nbrn = 24     # number of brands per market. if the numebr differs by market this requires some "accounting" vector

        # this vector relates each observation to the market it is in
        self.cdid = np.kron(np.array([i for i in range(self.nmkt)],ndmin=2).T,np.ones((self.nbrn,1)))
        self.cdid = self.cdid.reshape(self.cdid.shape[0]).astype('int')

        ## this vector provides for each index the of the last observation 
        ## in the data used here all brands appear in all markets. if this 
        ## is not the case the two vectors, cdid and cdindex, have to be   
        ## created in a different fashion but the rest of the program works fine.
        ##cdindex = [nbrn:nbrn:nbrn*nmkt]';
        self.cdindex = np.array([i for i in xrange((self.nbrn-1),self.nbrn*self.nmkt,self.nbrn)])

        ##% create weight matrix
        self.invA = np.linalg.inv(self.IV.T * self.IV)

        ## Logit results and save the mean utility as initial values for the search below

        ## compute the outside good market share by market
        temp = np.cumsum(self.s_jt)
        sum1 = temp[self.cdindex]
        sum1[1:] = np.diff(sum1)
        outshr = np.array([np.repeat(1-i,24) for i in sum1]).reshape(len(temp),1)

        y = np.log(self.s_jt) - np.log(outshr)
        mid = self.x1.T*self.IV*self.invA*self.IV.T
        t = np.linalg.inv(mid*self.x1)*mid*y
        self.d_old = self.x1*t
    ##    oldt2 = np.zeros(theta2.shape)
        self.d_old = np.exp(self.d_old)

        self.gmmvalold = 0
        self.gmmdiff = 1

    def init_theta(self):
        ## starting values for theta2. zero elements in the following matrix 
        ## correspond to coeff that will not be max over,i.e are fixed at zero.
        ## The rows are coefficients for the constant, price, sugar, and mushy
        ## dummy, respectively. The columns represent F, and interactions with
        ## income, income squared, age, and child.
        theta2w =  np.array([0.3302,5.4819,0,0.2037,0,
                     2.4526,15.8935,-1.2000,0,2.6342,
                     0.0163,-0.2506,0,0.0511,0,
                     0.2441,1.2650,0,-0.8091,0],ndmin=2)
        theta2w = theta2w.reshape((4,5))

        ##% create a vector of the non-zero elements in the above matrix, and the %
        ##% corresponding row and column indices. this facilitates passing values % 
        ##% to the functions below. %
        theti, thetj = list(np.where(theta2w!=0))
        theta2 = theta2w[np.where(theta2w!=0)]
        return [theta2,theti,thetj]

##        self.theta2w = pd.DataFrame(theta2w, index = ['constant','price','sugar','mushy'],
##                               columns = ['F','int.income','int.income_sq','int.age',
##                                          'int.child'])

    def gmmobj(self,theta2,theti,thetj):
        # compute GMM objective function
        delta = self.meanval(theta2,theti,thetj)
        
        ##% the following deals with cases were the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(delta)) == 1:
            f = 1e+10
        else:
            temp1 = self.x1.T*self.IV
            temp2 = delta.T*self.IV
            self.theta1 = np.linalg.inv(temp1*self.invA*temp1.T)*temp1*self.invA*temp2.T
            gmmresid = delta - self.x1*self.theta1
            temp1 = gmmresid.T*self.IV
            f = temp1*self.invA*temp1.T
        print 'gmm objective:',f[0,0]
        self.gmmvalnew = f
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        return(f[0,0])

    def meanval(self,theta2,theti,thetj):
        if self.gmmdiff < 1e-6:
            etol = self.etol = 1e-13
        elif self.gmmdiff < 1e-3:
            etol = self.etol = 1e-12
        else:
            etol = self.etol = 1e-9
        norm = 1
        avgnorm = 1
        i = 0
        theta2w = np.zeros((max(theti)+1,max(thetj)+1))
        for ind in range(len(theti)):
                theta2w[theti[ind],thetj[ind]]=theta2[ind]
        self.expmu = np.exp(self.mufunc(theta2w))
        while norm > etol:
            pred_s_jt = self.mktsh()
    ##        d_new = d_old + np.log(s_jt) - np.log(pred_s_jt)
            self.d_new = np.multiply(self.d_old,self.s_jt)/pred_s_jt
            t = np.abs(self.d_new - self.d_old)
            norm = np.max(t)
            avgnorm = np.mean(t)
            self.d_old = self.d_new
            i+=1
        print '# of iterations for delta convergence:',i   
        return np.log(self.d_new)

    def mufunc(self,theta2w):
        n,k = self.x2.shape
        j = theta2w.shape[1]-1
        mu = np.zeros((n,self.ns))
        for i in range(self.ns):
            v_i = self.vfull[:,np.arange(i,k*self.ns,self.ns)]
            d_i = self.dfull[:,np.arange(i,j*self.ns,self.ns)]
            temp = d_i*np.matrix(theta2w[:,1:(j+1)].T)
            mu[:,i]=(((np.multiply(self.x2,v_i)*theta2w[:,0])+np.multiply(self.x2,temp))*np.ones((k,1))).flatten()
        return mu

    def mktsh(self):
        # compute the market share for each product
        temp = self.ind_sh().T
        f = sum(temp)/float(self.ns)
        return f.T
    
    def ind_sh(self):
        eg = np.multiply(self.expmu,np.kron(np.ones((1,self.ns)),self.d_old))
        temp = np.cumsum(eg,0)
        sum1 = temp[self.cdindex,:]
        for i in range(1,sum1.shape[1]):
            sum1[1:sum1.shape[0],i] = np.diff(sum1[:,i].T).T
        denom1 = 1. / (1.+sum1)
        denom = denom1[self.cdid,:]
        return np.multiply(eg,denom)

if __name__ == '__main__':
    blp = BLP()
    init_theta = blp.init_theta()
    res = optimize.fmin(blp.gmmobj,init_theta[0],(init_theta[1],init_theta[2]))
