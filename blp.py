import os
import numpy as np
import scipy.io as io
import pandas as pd
from scipy import optimize

def gmmobj(theta2w,x1,IV,invA,delta_0,s_jt,v,demogr,x2,ns,nmkt,nbrn,cdindex,cdid):
    _theta2w = theta2w.reshape(4,5)
    # compute GMM objective function
    delta = meanval(delta_0,s_jt,_theta2w,vfull,dfull,x2,ns,nmkt,nbrn,cdindex,cdid)
    
    ##% the following deals with cases were the min algorithm drifts into region where the objective is not defined
    if max(np.isnan(delta)) == 1:
        f = 1e+10
    else:
        temp1 = x1.T*IV
        temp2 = delta.T*IV
        theta1 = np.linalg.inv(temp1*invA*temp1.T)*temp1*invA*temp2.T
        gmmresid = delta - x1*theta1
        temp1 = gmmresid.T*IV
        f = temp1*invA*temp1.T
    print 'gmm objective:',f[0,0]
    return(f[0,0])

def meanval(delta_0,s_jt,theta2w,vfull,dfull,x2,ns,nmkt,nbrn,cdindex,cdid):
    tol = 10e-8
    d_old = delta_0
    norm = 1
    avgnorm = 1
    i = 0
    expmu = np.exp(mufunc(theta2w,vfull,dfull,x2,ns,nmkt,nbrn))
    while norm > tol:
        pred_s_jt = mktsh(d_old,expmu,theta2w,vfull,dfull,x2,ns,nmkt,nbrn,cdindex,cdid)
##        d_new = d_old + np.log(s_jt) - np.log(pred_s_jt)
        d_new = np.multiply(d_old,s_jt)/pred_s_jt
        t = np.abs(d_new - d_old)
        norm = np.max(t)
        avgnorm = np.mean(t)
        d_old = d_new
        i+=1
    print '# of iterations for delta convergence:',i   
    return np.log(d_new)

def mufunc(theta2w,vfull,dfull,x2,ns,nmkt,nbrn):
    n,k = x2.shape
    j = theta2w.shape[1]-1
    mu = np.zeros((n,ns))
    thetw= np.matrix(theta2w)
    for i in range(ns):
        v_i = vfull[:,np.arange(i,k*ns,ns)]
        d_i = dfull[:,np.arange(i,j*ns,ns)]
        temp = d_i*thetw[:,1:(j+1)].T
        mu[:,i]=((np.multiply(x2,v_i)*thetw[:,0])+np.multiply(x2,temp)*np.ones((k,1))).flatten()
    return mu


def ind_sh(expmval,expmu,ns,cdindex,cdid):
    eg = np.multiply(expmu,np.kron(np.ones((1,ns)),expmval))
    temp = np.cumsum(eg,0)
    sum1 = temp[cdindex,:]
    for i in range(1,sum1.shape[1]):
        sum1[1:sum1.shape[0],i] = np.diff(sum1[:,i].T).T
    denom1 = 1. / (1.+sum1)
    denom = denom1[cdid,:]
    return np.multiply(eg,denom)

def mktsh(delta,expmu,theta2w,vfull,dfull,x2,ns,nmkt,nbrn,cdindex,cdid):
    # compute the market share for each product
    temp = ind_sh(delta,expmu,ns,cdindex,cdid).T
    f = sum(temp)/float(ns)
    return f.T


if __name__ == '__main__':
    ### Variable loadings and definitions. Much of the text is borrowed from
    ### http://emlab.berkeley.edu/users/bhhall/e220c/readme.html
    ps2 = io.loadmat('ps2.mat')
    iv = io.loadmat('iv.mat')
    
    # id variable in the format bbbbccyyq, where bbbb is a unique 4 digit
    # identifier for each brand, cc is a city code, yy is year and q is quarter.
    # All the other variables are sorted by date city brand. Phew.
    Id = ps2['id']
    
    brand = np.array([str(i[0])[0:4] for i in Id])
    city = np.array([str(i[0])[4:6] for i in Id])
    year = np.array([str(i[0])[6:8] for i in Id])
    quarter = np.array([str(i[0])[-1] for i in Id])
    
    IDEE = pd.DataFrame(np.array((brand,city,year,quarter)).T,columns=['brand','city','year','quarter'])
    
    # x1 variables enter the linear part of the estimation and include price
    # variable and 24 brand dummies. Each row corresponds to the equivalent
    # row in id.
    x1 = ps2['x1']
    
    # x2 variables enter the non-linear part and include a constant, price,
    # sugar content, a mushiness dummy. Each row corresponds to the equivalent
    # row in id.
    x2 = ps2['x2']
    
    # the market share of brand j in market t
    s_jt = ps2['s_jt']
    
    # id_demo is an id variable for the random draws (v) and the demographic
    # variables (demogr). Since these variables do not vary by brand they are
    # not repeated. The first observation here corresponds to the first market,
    # the second to the next 24 and so forth.
    Id_demo = ps2['id_demo']
    
    # Random draws. For each market 80 iid normal draws are provided.
    # They correspond to 20 "individuals", where for each individual
    # there is a different draw for each column of x2. The ordering is
    # given by id_demo. 
    v = ps2['v']

    # draws of demographic variables from the CPS for 20 individuals in each
    # market. The first 20 columns give the income, the next 20 columns the
    # income squared, columns 41 through 60 are age and 61 through 80 are a
    # child dummy variable. The ordering is given by id_demo. 
    demogr = ps2['demogr']

    IV = np.matrix(np.concatenate((iv['iv'][:,1:],x1[:,1:].todense()),1))

    ns = 20       # number of simulated "indviduals" per market 
    nmkt = 94     # number of markets = (# of cities)*(# of quarters)  
    nbrn = 24     # number of brands per market. if the numebr differs by market this requires some "accounting" vector

    # this vector relates each observation to the market it is in
    cdid = np.kron(np.array([i for i in range(nmkt)],ndmin=2).T,np.ones((nbrn,1)))
    cdid = cdid.reshape(cdid.shape[0]).astype('int')
    ##cdid = kron([1:nmkt]',ones(nbrn,1));

    ## this vector provides for each index the of the last observation 
    ## in the data used here all brands appear in all markets. if this 
    ## is not the case the two vectors, cdid and cdindex, have to be   
    ## created in a different fashion but the rest of the program works fine.
    ##cdindex = [nbrn:nbrn:nbrn*nmkt]';
    cdindex = np.array([i for i in xrange((nbrn-1),nbrn*nmkt,nbrn)] )

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
    theta2w = pd.DataFrame(theta2w, index = ['constant','price','sugar','mushy'],
                           columns = ['F','int.income','int.income_sq','int.age',
                                      'int.child'])

    ##% create a vector of the non-zero elements in the above matrix, and the %
    ##% corresponding row and column indices. this facilitates passing values % 
    ##% to the functions below. %
    ##theti, thetj = list(np.where(theta2w!=0))
    ##theta2 = theta2w[np.where(theta2w!=0)]

    ##horz = ['mean','sigma','income','income^2','age','child']
    ##vert = ['constant','price','sugar','mushy']
    ##% create weight matrix
    invA = np.linalg.inv(IV.T * IV)

    ## Logit results and save the mean utility as initial values for the search below

    ## compute the outside good market share by market
    temp = np.cumsum(s_jt)
    sum1 = temp[cdindex]
    sum1[1:] = np.diff(sum1)
    outshr = np.array([np.repeat(1-i,24) for i in sum1]).reshape(len(temp),1)

    y = np.log(s_jt) - np.log(outshr)
    mid = x1.T*IV*invA*IV.T
    t = np.linalg.inv(mid*x1)*mid*y
    mvalold = x1*t
##    oldt2 = np.zeros(theta2.shape)
    mvalold = np.exp(mvalold)

    vfull = np.array([np.repeat(i,24).reshape(80,24) for i in v]).swapaxes(1,2).reshape(2256,80)
    dfull = np.array([np.repeat(i,24).reshape(80,24) for i in demogr]).swapaxes(1,2).reshape(2256,80)
    
    res = optimize.fmin(gmmobj,theta2w,args=(x1,IV,invA,mvalold,s_jt,vfull,dfull,x2,ns,nmkt,nbrn,cdindex,cdid))
