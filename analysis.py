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

def VirtualCenter(U_int,V_int,orientation='EW'):
    """
    Finds the hodograph virtual center index
    """
    if orientation == 'EW':
        # Anticylonic case and EW section
        y_max = np.nanmax(np.abs(V_int))
        index = np.where(V_int == y_max)[0]
        if len(index)>1:
            index = index[0]
        else:
            index = None
    elif orientation == 'SN':
        x_max = np.nanmax(np.abs(U_int))
        index = np.where(U_int == x_max)[0]
        if len(index)>1:
            index = index[0]
        else:
            index = None
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
