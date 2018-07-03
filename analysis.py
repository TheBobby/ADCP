#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Functions to help with ADCP data analysis

__author__ = 'Antonin Affholder'

## Imports

from mpl_toolkits.basemap import Basemap
import numpy as np
import netCDF4 as nc
import datetime as dt

def ComputeATD(lon,lat,m = Basemap(projection='merc')):
    """
    Computes the along track distance between the given points
    """
    x,y = m(lon,lat)
    atd = np.cumsum(np.sqrt( (x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 ))
    atd = np.append(np.array([0]),atd)
    return(atd)

def RegularGrid(matrix,x,dx,y):
    """
    regularaizes the grid
    """
    xreg = np.arange(x[0] + dx/2,x[-1] - dx/2,dx)
    matrix_reg = np.full((len(xreg),len(y)),np.nan)
    for j in range(len(y)):
        for i in range(len(xreg)):
            val = xreg[i] #value to interpolate to
            valsup = val + dx/2
            valinf = val - dx/2
            #find the points laying in the interval
            indexes = np.where((x >= valinf) * (x <= valsup))
            if len(indexes) == 0:
                fill = np.nan
            elif np.sum(np.isfinite(matrix[indexes,j])) == 0:
                fill = np.nan
            else:
                fill = np.nanmean(matrix[indexes,j])
            matrix_reg[i,j] = fill
    return(matrix_reg,xreg)

def RegularLine(x,y,dx):
    xreg = np.arange(min(x) + dx/2,max(x) - dx/2,dx)
    matrix_reg = np.full(len(xreg),np.nan)
    for i in range(len(xreg)):
        val = xreg[i]
        valsup = val + dx/2
        valinf = val - dx/2
        #find the points laying in the interval
        indexes = np.where((x >= valinf) * (x <= valsup))
        if len(indexes) == 0:
            fill = np.nan
        elif np.sum(np.isfinite(y[indexes])) == 0:
            fill = np.nan
        else:
            fill = np.nanmean(y[indexes])
        matrix_reg[i] = fill
    return(matrix_reg,xreg)


def RegularGrid2D(matrix,x,y,dx,scale=1e3):
    """
    regularaizes the grid
    x and y have same size
    dx is in both directions
    Problems with empty squares
    """
    # x axis
    minxreg = np.floor(min(x)/scale)*scale
    maxxreg = np.ceil(max(x)/scale)*scale
    xbins = np.ceil((maxxreg-minxreg)/dx)
    xsreg = np.arange(minxreg + dx/2,minxreg + xbins*dx - dx/2,dx)
    # y axis
    minyreg = np.floor(min(y)/scale)*scale
    maxyreg = np.ceil(max(y)/scale)*scale
    ybins = np.ceil((maxyreg-minyreg)/dx)
    ysreg = np.arange(minyreg + dx/2,minyreg + ybins*dx - dx/2,dx)
    # New coordinates
    xreg = []
    yreg = []
    regmatrix = []
    for i in range(len(xsreg)):
        for j in range(len(ysreg)):
            # make the square in which we retrieve the data
            xregval = xsreg[i]
            yregval = ysreg[j]
            data_index = np.where(((x >= xregval - dx/2) * (x <= xregval + dx/2)) *
                              ((y >= yregval - dx/2) * (y <= yregval + dx/2)))[0]
            if len(data_index)>0:
                wcolumn = np.nanmean(matrix[data_index,:],axis=0)
                xreg.append(xregval)
                yreg.append(yregval)
                regmatrix.append(wcolumn)
    xreg = np.array(xreg)
    yreg = np.array(yreg)
    regmatrix = np.array(regmatrix)
    return(regmatrix,xreg,yreg)

def RegularGrid3D(matrix,x,y,dx,zbins,scale=1e3):
    """
    regularaizes the grid
    x and y have same size
    dx is in both directions
    """
    # x axis
    minxreg = np.floor(min(x)/scale)*scale
    maxxreg = np.ceil(max(x)/scale)*scale
    xbins = np.ceil((maxxreg-minxreg)/dx)
    xsreg = np.arange(minxreg + dx/2,minxreg + xbins*dx - dx/2,dx)
    # y axis
    minyreg = np.floor(min(y)/scale)*scale
    maxyreg = np.ceil(max(y)/scale)*scale
    ybins = np.ceil((maxyreg-minyreg)/dx)
    ysreg = np.arange(minyreg + dx/2,minyreg + ybins*dx - dx/2,dx)
    # New coordinates
    xreg = []
    yreg = []
    regmatrix = np.full((len(xsreg),len(ysreg),zbins),np.nan)
    for i in range(len(xsreg)):
        for j in range(len(ysreg)):
            # make the square in which we retrieve the data
            xregval = xsreg[i]
            yregval = ysreg[j]
            data_index = np.where(((x >= xregval - dx/2) * (x <= xregval + dx/2)) *
                              ((y >= yregval - dx/2) * (y <= yregval + dx/2)))[0]
            if len(data_index)>0:
                wcolumn = np.nanmean(matrix[data_index,:],axis=0)
                regmatrix[i,j,:] = wcolumn
    return(regmatrix,xsreg,ysreg)

def BoxCarFilter(M,bins_x,bins_y):
    """
    Boxcar filter, or blur
    Every point is the average of the box x_len,ylen
    M is the field matrix of size = (len(X),len(Y))
    bins_x is the HALF of the horizontal size of the box in bins
    bins_y is the HALF of the vertical size of the box in bins
    """
    xl,yl = M.shape
    boxdim = 2*bins_x*2*bins_y
    Mp = np.zeros((xl,yl))
    for i in range(xl):
        for j in range(yl):
            xmin = i - bins_x
            ymin = j - bins_y
            xmax = i + bins_x
            ymax = j + bins_y

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > xl:
                xmax = -1
            if ymax > yl:
                ymax = -1
            box = M[xmin : xmax, ymin : ymax]
            if np.sum(np.isfinite(box)) < boxdim/2:
                avg = np.nan
            else:
                avg = np.nanmean(box)
            Mp[i,j] = avg
    return(Mp)

def BoxCarFilter3D(M,x,y,z,xdist,ydist,zdist):
    """
    This is not supposed to work
    """
    xl,yl,zl = M.shape
    Mp = np.zeros((xl,yl,zl)) # should be nans

    dx = xdist/2
    dy = ydist/2
    dz = zdist/2
    xbins = np.ceil(xdist/np.nanmean(x[1:] - x[:-1]))
    ybins = np.ceil(ydist/np.nanmean(y[1:] - y[:-1]))
    zbins = np.ceil(zdist/np.nanmean(z[1:] - z[:-1]))
    boxsize = xbins*ybins*zbins


    for i in range(xl):
        for j in range(yl):
            for k in range(zl):

                xval = x[i]
                yval = y[j]
                zval = z[k]

                indexes = np.where( ( x >= xval - dx)*( x <= xval + dx)*
                                    ( y >= yval - dy)*( y <= yval + dy)*
                                    ( z >= zval - dz)*( z <= zval + dz))
                if np.sum(np.isfinite(M[indexes])) < boxsize/6:
                    val = np.nan
                else:
                    val = np.nanmean(M[indexes])
                Mp[i,j,k] = avg
    return(Mp)

def BoxCarFilter2(M,x,y,z,xdist,ydist,zdist):
    """
    Boxcar filter, or blur
    Taking into account that the data is distributed in a 3D space
    means the spheroid
    x y same size but not M
    M is 2D

    """
    X = []
    Y = []
    Z = []
    # Vl = []
    for i in range(len(x)):
        for j in range(len(z)):
            X.append(x[i])
            Y.append(y[i])
            Z.append(z[j])
    #         Vl.append(V[i,j])
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    # Vl = np.array(Vl)
    Mf = M.flatten()
    xl,zl = M.shape

    dx = xdist/2
    dy = ydist/2
    dz = zdist/2

    xbins = np.ceil(xdist/np.nanmean(x[1:] - x[:-1]))
    ybins = np.ceil(ydist/np.nanmean(y[1:] - y[:-1]))
    zbins = np.ceil(zdist/np.nanmean(z[1:] - z[:-1]))
    boxsize = xbins*zbins

    Mp = np.full(len(X),np.nan)
    for i in range(len(X)):
        xval = X[i]
        yval = Y[i]
        zval = Z[i]

        indexes = np.where( ( X >= xval - dx)*( X <= xval + dx)*
                            ( Y >= yval - dy)*( Y <= yval + dy)*
                            ( Z >= zval - dz)*( Z <= zval + dz))[0]
        if np.sum(np.isfinite(Mf[indexes])) < boxsize/8:
            val = np.nan
        else:
            val = np.nanmean(Mf[indexes])
        Mp[i] = val
    Mpuf = np.reshape(Mp,(xl,zl))
    return(Mpuf)


def FindMaxMin(V,atd,depths,depthlim = [0,3000]):
    """

    """
    atd_min = []
    atd_max = []
    dpt = []

    depthmin = depthlim[0]
    depthmax = depthlim[1]

    subdepths = depths[(depths > depthmin) * (depths < depthmax)]

    for i in range(len(subdepths)):
        index = np.where(depths == subdepths[i])
        vs = V[:,index]
        if np.sum(np.isfinite(vs)) > 0:
            min_index = np.where(vs == np.nanmin(vs))[0][0]
            max_index = np.where(vs == np.nanmax(vs))[0][0]
            dpt.append(subdepths[i])
            atd_min.append(atd[min_index])
            atd_max.append(atd[max_index])
    atd_min = np.array(atd_min)
    atd_max = np.array(atd_max)
    dpt = np.array(dpt)
    return(atd_min,atd_max,dpt)

def TrackSSH(lon,lat,SSH,date,mid=True):
    """
    Gives the SLA measured on the ship's track at date
    SSH is a netcdf file
    date is datetime date
    """
    tunits = SSH['time'].units
    time = nc.num2date(SSH['time'][:],tunits)
    if mid: #12hrs offset between date and dates in time
        time_index = np.where(time == date + dt.timedelta(hours=12))[0][0]
    else:
        time_index = np.where(time == date)[0][0]
    ssh = []
    xstep = SSH['longitude'].step
    ystep = SSH['latitude'].step
    hxstep = np.round(xstep/2,3)
    hystep = np.round(ystep/2,3)
    ssh_xaxis = SSH['longitude'][:]
    ssh_yaxis = SSH['latitude'][:]
    zos = SSH['zos'][time_index,:,:]
    for i in range(len(lon)):
        longitude = lon[i]
        latitude = lat[i]
        lg_idx = np.where((ssh_xaxis >= longitude - hxstep)*(ssh_xaxis <= longitude + hxstep))[0][0]
        lt_idx = np.where((ssh_yaxis >= latitude - hystep)*(ssh_yaxis <= latitude + hystep))[0][0]
        ssh.append(zos[lt_idx,lg_idx])
    ssh = np.array(ssh)
    return(ssh)

def VirtualCenter(v,orientation='EW',deltat=10):
    """
    Finds the hodograph virtual center index
    """
    #U_int = np.nancumsum(u)*deltat
    V_int = np.nancumsum(v)*deltat
    if orientation == 'EW':
        # Anticylonic case and EW section
        y_max = np.nanmax(np.abs(V_int))
        index = np.where(V_int == y_max)[0]
        if len(index)>0:
            index = index[0]
        else:
            index = False
    elif orientation == 'SN':
        x_max = np.nanmax(np.abs(U_int))
        index = np.where(U_int == x_max)[0]
        if len(index)>0:
            index = index[0]
        else:
            index = False
    return(index)



def RadialVorticity(v,x):
    '''
    Poor man's vorticity
    '''
    zeta = []
    for j in range(len(x)):
        Vs = v[j,:]
        Bs = Vs/x[j]
        zetas = 2*np.arctan(Bs)
        zeta.append(zetas)
    zeta = np.array(zeta)
    return(zeta)
