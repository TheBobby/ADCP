#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Functions to help with plots of ADCP data

__author__ = 'Antonin Affholder'

import matplotlib.pyplot as plt
import numpy as np
import analysis


def Hodograph(ax,xlim):
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
    x_ticks = np.arange(0.3,xlim + 0.3,2)*np.cos(np.pi/6)
    y_ticks = np.arange(0.3,xlim + 0.3,2)*np.sin(np.pi/6)
    ticks = np.array(np.arange(0,xlim,2),dtype=str)
    # Plot the ticks
    for j in range(len(x_ticks)):
        ax.text(x_ticks[j],y_ticks[j],ticks[j])
    # Annotate the axes
    ax.text(0,-xlim - 1.5,'Eastward distance (km)',fontsize=16,horizontalalignment='center',verticalalignment='center')
    ax.text(-xlim - 1.5,0,'Northward distance (km)',fontsize=16,rotation=90,horizontalalignment='center',verticalalignment='center')
    # Make the grid
        # Lines
    angle = np.pi/6
    for i in range(6):
        angle = i*np.pi/6
        angle2 = angle+np.pi
        angles = np.array([angle,angle2])
        x = np.cos(angles)*xlim
        y = np.sin(angles)*xlim
        xtext = np.cos(angles)*(xlim + .5)
        ytext = np.sin(angles)*(xlim + .5)
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
    for i in range((xlim//2)+1):
        circle = plt.Circle((0,0),i*2,fill=False,color='grey',alpha=0.4)
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
        y_max = np.nanmax(y)
        index = np.where(y == y_max)[0]
        if len(index)>1:
            index = index[0]
        x_center = x[index]
        y_center = y_max
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

def PlotADCP(ax,atd,depths,V):
    """
    Still unfinished with flexibility of reg and filt
    TODO adapt to filtered and regular data
    Returns km
    """
    X,Y = np.meshgrid(atd/1000,depths)
    lvls = np.round(np.arange(-0.8,0.9,0.1),1)
    lvls = np.delete(lvls,8)
    ax.contourf(X,Y,np.transpose(V),levels=lvls)
    cont = ax.contour(X,Y,np.transpose(V),levels = lvls,linewidths=1.5)
    ax.clabel(cont, cont.levels,fmt='%0.1f',fontsize=8, inline=1,colors='k')
    lvl0 = [0]
    ax.contour(X,Y,np.transpose(V),levels=lvl0,colors='black',linewidths=2)

def PlotMaxMin(ax,V,atd,depths):
    """
    """
    atd_min,atd_max,dpt = analysis.FindMaxMin(V,atd,depths)
    ax.plot(atd_min/1000,dpt,'k--')
    ax.plot(atd_max/1000,dpt,'k--')
