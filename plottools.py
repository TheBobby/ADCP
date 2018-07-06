#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Functions to help with plots of ADCP data

__author__ = 'Antonin Affholder'

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import analysis
import matplotlib.tri as tri
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc

### Defining global colormap for ADCP Plotting
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = -0.5, vmax = 0.5)
levels = np.round(np.arange(-0.5,0.55,.05),2)
levels2 = np.delete(levels,np.where(levels==0))

def ADCP_surf(ax,eddy,eddies,ADCPdata,SSHdata,contours,detections,grid=2):
    '''
    Returns a figure with the quiver plot of velocities at required depth along cruise track on selected eddy
    eddie is a string
    ADCPdata is a dict of the good shape
    depth is a float or integer
    '''
    ## Useful variables
    ID = eddies[eddy]['ID']
    boxx = eddies[eddy]['box_X']
    boxy = eddies[eddy]['box_Y']
    lgmin = min(boxx)
    lgmax = max(boxx)
    ltmin = min(boxy)
    ltmax = max(boxy)
    index_dtc = np.where(detections['IDs'] == ID)
    date = dt.datetime.fromordinal(detections['Dates'][index_dtc][0])
    dptmax = 150
    dptmin = 50
    ## Retrieving ADT field
    date = nc.date2num(date,'hours since 1950-01-01 12:00:00')
    index = np.where(file['time'][:] == date)[0][0]
    sshlon = file['longitude'][:]
    sshlat = file['latitude'][:]
    adt = file['zos'][index,:,:]
    X,Y = np.meshgrid(sshlon,sshlat)
    ## Center of detection
    xcenter = detections['Xs'][index_dtc]
    ycenter = detections['Ys'][index_dtc]
    ## Retrieving contour
    contour = contours[ID]
    xc = contour['X']
    yc = contour['Y']
    ## Retrieving ADCP data
    longitudes = ADCPdata['longitudes']
    latitudes = ADCPdata['latitudes']
    dindexes = (ADCPdata['depths'] < dptmax)*ADCPdata['depths'] > dptmin
    pindexes = (longitudes < lgmax)*(longitudes > lgmin)*(latitudes < ltmax)*(latitudes > ltmin)
    U = ADCPdata['U'][pindexes,indexes].mean(axis=1)
    V = ADCPdata['V'][pindexes,indexes].mean(axis=1)
    ## Setting up the projection
    m = Basemap(projection='merc',llcrnrlat=ltmin,urcrnrlat=ltmax,llcrnrlon=lgmin,urcrnrlon=lgmax,resolution='c')
    Xc,Yc = m(xc,yc)
    Xcenter,Ycenter = m(xcenter,ycenter)
    x,y = m(longitudes,latitudes)
    ## Plotting
    # ADT field
    mesh = m.pcolormesh(X,Y,adt,cmap = plt.cm.jet,latlon=True)
    mesh.set_clim(vmin = -0.05, vmax = 0.4)
    cbar = plt.colorbar(mesh,ticks = np.arange(-0.05,0.41,.1))
    cbar.set_label('ADT (m)')
    # Contour
    ax.plot(Xc,Yc)
    # Center
    ax.plot(Xcenter,Ycenter,'*',ms=5)
    # Quiver
    Q = ax.quiver(x,y,U,V,scale=5,pivot='tail',width=0.001)
    xleg = lgmin + 0.7
    yleg = ltmin + 0.1
    x,y = m(xleg,yleg)
    ax.quiverkey(Q,x,y,1, '1 m/s', coordinates='data')

    ax.plot(x,y,'-k',linewidth=0.2)
    # Meridians and parallels
    parallels=np.arange(ltmin,ltmax,grid)
    meridians=np.arange(lgmin,lgmax,grid)
    m.drawparallels(parallels,labels=[1,0,0,0],color='grey')
    m.drawmeridians(meridians,labels=[0,0,0,1],color='grey')
    # Coastlines
    m.drawcoastlines()
    # Title
    title = 'Velocity averaged between ' + str(dptmin) + ' and ' + str(dptmax) + 'm around eddy ' + eddy
    ax.title(title)


def Hodograph(ax,xlim,xtext=True,ytext=True,grid=2):
    """
            *** Function Hodograph ***
    Makes the suitable background for a hodograph plot
            *** Arguments ***
    - ax is an pyplot Axes instance that should be empty
    - xlim is the maximum extension of the plot (ray)
            *** Outputs ***
    No outputs, works on the Axes instance directly
            *** Remarks ***
    Set xlim in km
    """
    ## Initialization
    # Set axis limits, in order to see circle aspect MUST be set to equal
    ax.set_xlim(-xlim,xlim)
    ax.set_ylim(-xlim,xlim)
    ax.set_aspect('equal')
    # Make the axis disappear
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # Create ticks for the new axis and set their position
    x_ticks = np.arange(0.3,xlim + 0.3,grid)*np.cos(np.pi/6)
    y_ticks = np.arange(0.3,xlim + 0.3,grid)*np.sin(np.pi/6)
    ticks = np.array(np.arange(0,xlim,grid),dtype=str)
    # Plot the ticks
    for j in range(len(x_ticks)):
        ax.text(x_ticks[j],y_ticks[j],ticks[j])
    # Annotate the axes
    if xtext:
        ax.text(0,-xlim - xlim/6,'Eastward distance (km)',fontsize=16,horizontalalignment='center',verticalalignment='center')
    if ytext:
        ax.text(-xlim - xlim/6,0,'Northward distance (km)',fontsize=16,rotation=90,horizontalalignment='center',verticalalignment='center')
    # Make the grid
        # Lines
    angle = np.pi/6
    for i in range(6):
        angle = i*np.pi/6
        angle2 = angle+np.pi
        angles = np.array([angle,angle2])
        x = np.cos(angles)*xlim
        y = np.sin(angles)*xlim
        xtext = np.cos(angles)*(xlim + xlim/12)
        ytext = np.sin(angles)*(xlim + xlim/12)
        ax.plot(x,y,'grey',alpha=0.3)
        deg = int(np.round(np.rad2deg(angle-np.pi/2)))
        deg2 = int(np.round(np.rad2deg(angle2-np.pi/2)))
        if deg < 0:
            deg = deg + 360
        if deg2 < 0:
            deg2 = deg2 + 360
        ax.text(x = xtext[0],y = ytext[0],s = str(deg)+'°',horizontalalignment='center',verticalalignment='center')
        ax.text(x = xtext[1],y = ytext[1],s = str(deg2)+'°',horizontalalignment='center',verticalalignment='center')

        # Angle ticks
    for i in range(len(x_ticks)):
        circle = plt.Circle((0,0),i*grid,fill=False,color='grey',alpha=0.4)
        ax.add_artist(circle)
    circle = plt.Circle((0,0),xlim,fill=False,color='k',alpha=1)
    ax.add_artist(circle)

def PlotHodograph(ax,U,V,deltat,legend=True,orientation='EW',type='A'):
    """
            *** Function PlotHodograph ***
    Adds hodograph plot to the hodograph background
            *** Arguments ***
    - ax is an pyplot Axes instance that should contain the proper background
    produced with analysis.Hodograph
    - U and V are 1D arrays containing the measured velocities
    - deltat is the time between each velocity sampling
    * kwargs *
        - legend: plots legend if True.
            default = True
        - orientation: direction of cruisetrack, zonal 'EW' or meridional 'SN'
            default = 'EW'
        - type: eddy type, anticyclonic 'A' or cyclonic 'C'
            default = 'A'
            *** Outputs ***
    No outputs, works on the Axes instance directly
            *** Remarks ***
    """
    ## Plot the data
    # Make hodograph from velocities
    x = (np.nancumsum(U)*deltat)/1000 # /1000 to get km
    y = (np.nancumsum(V)*deltat)/1000 # /1000 to get km
    # Plot the time integration
    hodograph = ax.plot(x,y)
    # Plot the first point
    first_point, = ax.plot(x[0],y[0],'ko',ms=10,label='First data point')
    # Plot the last point
    last_point, = ax.plot(x[-1],y[-1],'kd',ms=10,label='Last data point')
    # Find the virtual center
    if orientation == 'EW' and type == 'A':
        # Anticylonic case and EW section
        y_max = np.nanmax(np.abs(y))
        index = np.where(np.abs(y) == y_max)[0]
        if len(index)>1:
            index = index[0]
        x_center = x[index]
        y_center = y[index]
    elif orientation == 'SN' and type == 'A':
        x_max = np.nanmax(x)
        index = np.where(x == x_max)[0]
        if len(index)>1:
            index = index[0]
        y_center = y[index]
        x_center = x_max
    # Plot the virtual center point
    center, = ax.plot(x_center,y_center,'k*',ms=10,label='Virtual center')
    # Legend
    if legend:
        ax.legend(handles = [first_point,last_point,center],bbox_to_anchor=(1.1, 1))

def PlotADCP(ax,atd,depths,V,levels=levels,levels2=levels2,cmap=cmap,norm=norm):
    """
    Still unfinished with flexibility of reg and filt
    TODO adapt to filtered and regular data
    Returns km
    levels can be
    """
    X = []
    Z = []
    for i in range(len(atd)):
        for j in range(len(depths)):
            X.append(atd[i])
            Z.append(depths[j])
    X = np.array(X)
    Z = np.array(Z)
    Vf = V.flatten()
    # Regular grid
    xi = np.linspace(min(atd),max(atd), len(atd))
    zi = np.linspace(min(depths), max(depths), len(depths))
    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(X, Z)
    interpolator = tri.LinearTriInterpolator(triang, Vf)
    Xi, Zi = np.meshgrid(xi, zi)
    Vi = interpolator(Xi, Zi)
    ax.contourf(Xi,Zi,Vi,levels=levels,cmap=cmap,norm=norm)
    cont = ax.contour(Xi,Zi,Vi,levels = levels2,linewidths=0.5,colors='black',linestyles='dashed')
    ax.clabel(cont,cont.levels,fmt='%1.2f',fontsize=8, inline=1,colors='k')
    lvl0 = [0]
    ax.contour(Xi,Zi,Vi,levels=lvl0,colors='black',linewidths=2)

def PlotMaxMin(ax,V,atd,depths):
    """
    """
    atd_min,atd_max,dpt = analysis.FindMaxMin(V,atd,depths)
    ax.plot(atd_min/1000,dpt,'k--')
    ax.plot(atd_max/1000,dpt,'k--')
