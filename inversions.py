#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.ma as ma

def Simulate(x_center,y_center,x_me,y_me,omega=np.pi*1e-6,type='A',fmt='AN'):
    """
            *** Function Simulate ***
    Simulates the velocities of a solid body rotation eddy
            *** Arguments ***
    - x_center, y_center are floats, coordinates of the center of the eddy to simulate
    - x_me, y_me are the coordinates of the points to simulate
    * kwargs *
        - omega is solid body rotation angular velocity (rad.s-1)
            default = pi*1e-6
        - type: eddy type, anticyclonic 'A' or cyclonic 'C'
            default = 'A'
        - fmt : output format 'AN' is angle norm (faster) 'UV' is U V
            *** Outputs ***
    - u,v in the same dimension as x_me and y_me, simulated velocities
            *** Remarks ***
    """
    r = np.sqrt( (x_center - x_me)**2 + (y_center - y_me)**2 )
    # angle = np.arctan2(y_me-y_center,x_me-x_center) #convention of artcan2 inverts y and x coordinates
    angle = np.angle((x_me - x_center) + (y_me - y_center)*1j,deg=False)
    norm = r * np.tan(omega)
    if type == 'A':
        offset = np.pi/2
    elif type == 'C':
        offset =  - np.pi/2
    angle = angle + offset
    if fmt == 'UV':
        u = norm * np.cos(angle)
        v = norm * np.sin(angle)
        return(u,v)
    elif fmt == 'AN':
        return(angle,norm)

def Rankine(r,par):
    R,V = par
    v = np.zeros(r.shape)
    v[np.abs(r) <= R] = V*r[np.abs(r) <= R]/R
    v[np.abs(r) > R] = V*R/r[np.abs(r) > R]
    return(v)

def RankineErr(par,r,vm):
    """
    par is (R,V)
    """
    #rR,rV = par
    rVs = Rankine(r,par)
    rr = np.nansum(np.sqrt((vm - rVs)**2))/np.sqrt(2*np.sum(np.isfinite(vm)))
    return(rr)

def SimulateRankine(center,par,x,y,type='A',cut='Z',fmt='UV'):
    """
    Simulates a Rankine vortex sampled on x and y
    optional:
        type: type of eddy (default A for anticyclonic, can be C)
        cut: orientation of the cut (default Z for Zonal M for meridional)
        fmt: output format, can be UV or AN for angles and norms
    """
    xc,yc = center
    #R,V = par
    if cut == 'Z':
        coord = x
        centercoord = xc
    elif cut == 'M':
        coord = y
        centercoord = yc
    else:
        raise ValueError('''Unexpected value for cut. must be either 'Z' or 'M' ''',cut)
    rs = np.sqrt((xc- x)**2 + (yc - y)**2)
    angles = np.angle((x - xc) + (y - yc)*1j,deg=False)
    norms = Rankine(rs,par)
    if type == 'A':
        offset = np.pi/2
    elif type == 'C':
        offset = -np.pi/2
    else:
        raise ValueError('''Unexpected value for type, should be 'A' or 'C' got ''',type)
    angles = angles + offset
    if fmt == 'UV':
        u = norms * np.cos(angles)
        v = norms * np.sin(angles)
        return(u,v)
    elif fmt == 'AN':
        return(angles,norms)
    else:
        raise ValueError('''Unexpected value for fmt, should be 'AN' or 'UV' got ''',fmt)


def R_lin(size,data,coords,zero):
    """
    Gives the r squared for the reg around center of size size
    """
    if np.isfinite(zero):
        subdata = data[(coords > zero - size)*(coords < zero + size)]
        subcoords = coords[(coords > zero - size)*(coords < zero + size)]
        fidx = np.where(np.isfinite(subdata))[0]
        if len(fidx) > 0:
            fdata = subdata[fidx]
            fcoords = subcoords[fidx]
            res = stats.linregress(fcoords,fdata)
            coeffs = np.array([res[0],res[1]])
            omega = np.abs(np.arctan(coeffs[0]))
            rs = res[2]**2
            p = res[3]
        else:
            rs = np.nan
            p = np.nan
            omega = np.nan
            coeffs = np.array([np.nan,np.nan])
    else:
        rs = np.nan
        p = np.nan
        omega = np.nan
        coeffs = np.array([np.nan,np.nan])
    return(rs,p,omega,coeffs)

def SBExtension(data,xcoords,deltat,sizemax = 200e3):
    """
    data is 1D
    coords is coords
    to fix: allow asymmetry around center
    """
    Ss = np.arange(1e3,sizemax+1e3,1e3)
    lgth = len(Ss)
    Rs = []
    Ps = []
    Os = []
    Cs = []

    if np.sum(np.isfinite(data)) == 0:
        Rs = np.full(lgth,np.nan)
        Ps = np.full(lgth,np.nan)
        Os = np.full(lgth,np.nan)
        Cs = np.full(lgth,np.nan)
        zero = np.nan
    else:
        vmint = np.nancumsum(data)*deltat
        ymax = np.nanmax(vmint)
        vc_idx = np.where(vmint == ymax)[0]
        zero = xcoords[vc_idx]
        if len(zero) == 0:
            zero = np.nan
        elif len(zero) > 0:
            zero = zero[0]
        for size in Ss:
            r,p,o,coeffs = R_lin(size,data,xcoords,zero)
            Os.append(o)
            Rs.append(r)
            Ps.append(p)
            Cs.append(coeffs)
    Ss = np.array(Ss)
    Rs = np.array(Rs)
    Ps = np.array(Ps)
    Os = np.array(Os)
    Cs = np.array(Cs)
    return(Ss,Rs,Ps,Os,Cs,zero)

def AngularError(coords,xm,ym,cos_m,sin_m):
    xc = coords[0]
    yc = coords[1]
    angles_s,norms_s = Simulate(xc,yc,xm,ym,omega=1e-5,fmt='AN')
    cos_s = np.cos(angles_s)
    sin_s = np.sin(angles_s)
    rr = np.sqrt((sin_m - sin_s)**2 + (cos_m - cos_s)**2)
    N = np.sum(np.isfinite(rr))
    if N == 0:
        rr = np.nan
    else:
        rr = np.nansum(rr)/np.sqrt(2*N)
    return(rr)

def Error(pos,xm,ym,um,vm):
    """
    Computes non only the angular error but the absolute distance in the complex plane
    """
    xc,yc,o = pos
    us,vs = Simulate(xc,yc,xm,ym,omega=o,fmt='UV')
    rr = np.sqrt((um - us)**2 + (um - us)**2)
    N = np.sum(np.isfinite(rr))
    if N == 0:
        rr = np.nan
    else:
        rr = np.nansum(rr)/np.sqrt(2*N)
    return(rr)

def MapError(xplore,yplore,xm,ym,cos_m,sin_m,mask=None):
    """
    """
    if mask is None:
        msk = np.array([True for i in range(len(xm))],dtype='bool')
    else :
        msk = ~mask
    Merr = np.full((len(xplore),len(yplore)),np.nan)
    for i in range(len(xplore)):
        for j in range(len(yplore)):
            xi = xplore[i]
            yj = yplore[j]
            rr = AngularError([xi,yj],xm[msk],ym[msk],cos_m[msk],sin_m[msk])
            Merr[i,j] = rr
    return(Merr)

def SolidBodyCorrelation(U,V,atd,depths,sizemax=200e3,deltat=10):
    """
    Extracts the SBR component from the data
    Problem is that if the boat track is not linear it could artificially reduce the SBRC extension..
    Also, coded only for A eddies
    """
    # Init
    Rvals = []
    Pvals = []
    Zeros = []
    # For every depth compute different distances from the center correlations with lines
    for i in range(len(depths)):
        um = U[:,i]
        vm = V[:,i]
        size,rval,pval,omega,coeffs,zero = SBExtension(vm,atd,deltat,sizemax=sizemax)
        Rvals.append(rval)
        Pvals.append(pval)
        Zeros.append(zero)
    ##
    Rvals = np.array(Rvals)
    Pvals = np.array(Pvals)
    Zeros = np.array(Zeros)
    ## putting everything on nice matrixes
    plus = np.array([Zeros[i] + size for i in range(len(depths))])
    moins = np.array([Zeros[i] - size for i in range(len(depths))])
    moins = np.fliplr(moins)
    faisceau = np.append(moins,plus,axis=1)
    ZZ2 = np.repeat(depths,len(size)*2).reshape((len(depths),len(size)*2))
    slavR = np.fliplr(Rvals)
    Rvals_sim = np.append(slavR,Rvals,axis=1)
    slavP = np.fliplr(Pvals)
    Pvals_sim = np.append(slavP,Pvals,axis=1)
    return(faisceau,ZZ2,Rvals_sim,Pvals_sim)

def SBRCindex(Rvals,atd,faisceau,depths,thresh=0.8):
    ## Extract solid body from the data
    indexes_x = np.array([])
    indexes_z = np.array([])
    for i in range(len(depths)):
        faisc = faisceau[i,Rvals[i,:] > thresh]
        if len(faisc) > 0:
            atd_inf = np.nanmin(faisc)
            atd_sup = np.nanmax(faisc)
            indexes_to_mask = np.where((atd < atd_inf)+(atd > atd_sup))[0]
            dindex = np.array([i]*len(indexes_to_mask))
            indexes_x = np.append(indexes_x,indexes_to_mask)
            indexes_z = np.append(indexes_z,dindex)
    indexes = (np.array(indexes_x,dtype=int),np.array(indexes_z,dtype=int))
    return(indexes)
