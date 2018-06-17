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
    ssh_xaxis = SSH['longitude'][:]
    ssh_yaxis = SSH['latitude'][:]
    zos = SSH['zos'][time_index,:,:]
    for i in range(len(lon)):
        longitude = lon[i]
        latitude = lat[i]
        lg_idx = np.where((ssh_xaxis >= longitude - xstep/2)*(ssh_xaxis <= longitude + xstep/2))[0][0]
        lt_idx = np.where((ssh_yaxis >= latitude - ystep/2)*(ssh_yaxis <= latitude + ystep/2))[0][0]
        ssh.append(zos[lt_idx,lg_idx])
    ssh = np.array(ssh)
    return(ssh)


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

def Simulate(x_center,y_center,x_me,y_me,omega=np.pi*1e-6,type='A'):
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
            *** Outputs ***
    - u,v in the same dimension as x_me and y_me, simulated velocities
            *** Remarks ***
    """
    r = np.sqrt( (x_center - x_me)**2 + (y_center - y_me)**2 )
    angle = np.arctan2(y_me-y_center,x_me-x_center) #convention of artcan2 inverts y and x coordinates
    norm = r * np.tan(omega)
    if type == 'A':
        u = norm * np.cos(angle + np.pi/2)
        v = norm * np.sin(angle + np.pi/2)
    else:
        u = norm * np.cos(angle - np.pi/2)
        v = norm * np.sin(angle - np.pi/2)
    return(u,v)

def FindCenter(U,V,lon,lat,lon_c_prior,lat_c_prior,complete=False,m=Basemap(projection='merc')):
    '''
    to refine to look only around the zone where the center might be
    U and V are 1D arrays
    '''
    #projection
    x,y = m(lon,lat)
    x_c_prior,y_c_prior = m(lon_c_prior,lat_c_prior)

    #Grid
    x_box = np.arange(x_c_prior - 20000,x_c_prior + 20100,500)
    y_box = np.arange(y_c_prior - 20000,y_c_prior + 20100,500)
    X_box,Y_box = np.meshgrid(x_box,y_box)
    S = np.zeros((len(y_box),len(x_box)))
    S_std = np.zeros((len(y_box),len(x_box)))

    #Compute scores
    for j in range(len(x_box)):
        for i in range(len(y_box)):
            x_center = x_box[j]
            y_center = y_box[i]
            u,v = Simulate(x_center,y_center,x,y,omega=5.7e-6)
            angles = np.arccos((U * u + V * v)/(np.sqrt(U**2 + V**2) * np.sqrt(u**2 + v**2)))
            score = np.nanmean(angles)
            std = np.nanstd(angles)
            S[i,j] = score
            S_std[i,j] = std
    # find center
    x_index = np.where(S == np.nanmin(S))[0][0]
    y_index = np.where(S == np.nanmin(S))[1][0]
    x_center = X_box[x_index,y_index]
    y_center = Y_box[x_index,y_index]
    lon_center,lat_center = m(x_center,y_center,inverse=True)
    if complete:
        return(lon_center,lat_center,x_center,y_center,S,S_std)
    else:
        return(lon_center,lat_center)

def FindOmega(U,V,lon,lat,lon_center,lat_center):
    '''
    finds omega
    '''
    omegas = np.arange(0,2*np.pi/1000000,0.0000001)
    scores = []
    m = Basemap(projection='merc')
    x,y = m(lon,lat)
    x_center,y_center = m(lon_center,lat_center)
    for i in range(len(omegas)):
        u,v = Simulate(x_center,y_center,x,y,omega=omegas[i])
        score = np.nansum(np.sqrt((u - U)**2 + (v - V)**2))
        scores.append(score)
    scores = np.array(scores)
    omega = omegas[np.where(scores == np.nanmin(scores))][0]
    score = np.nanmin(scores)
    return(omega,score)

def FindCenter2(eddies,eddy,m=Basemap(projection='merc')):
    '''
    level 2 function
    '''
    # Data
    ADCP = eddies[eddy]['ADCP']
    lon = ADCP['longitudes']
    lat = ADCP['latitudes']
    depths = ADCP['depths']
    lon_c_prior = eddies[eddy]['X_RS_center']
    lat_c_prior = eddies[eddy]['Y_RS_center']
    #depths mean 350 first meter, to change
    index = np.where(depths < 350)[0]
    U_me = np.nanmean(ADCP['U'][:,index],axis=1)
    V_me = np.nanmean(ADCP['V'][:,index],axis=1)
    # Find center
    lon_center,lat_center = FindCenter(U_me,V_me,lon,lat,lon_c_prior,lat_c_prior)
    omega,score = FindOmega(U_me,V_me,lon,lat,lon_center,lat_center)
    return(lon_center,lat_center,omega)
