#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Functions to help with ADCP data manipulation and plots
### The philosophy of these function is that they only require the ADCP data
### For more high level use of the data, see --INSERT--

## Imports
import matplotlib.pyplot as plt
import matplotlib.path as pth
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.basemap import Basemap
#from sklearn import linear_model

#### Subsetting the data
# This will only work on raw formatted data that was produced with functions from
# the ADCP_file package

def Hodograph(U,V,deltat,orientation='EW',eddy='A'):
    '''
    Makes an hodograph
    '''
    ## Initialization
    # Create figuregg
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    # Set axis limits, in order to see circle aspect MUST be set to equal
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    ax.set_aspect('equal')
    # Make the axis disappear
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # Create ticks for the new axis and set their position
    x_ticks = np.arange(0.2,8,2.2)*np.cos(np.pi/6)
    y_ticks = np.arange(0.2,8,2.2)*np.sin(np.pi/6)
    ticks = np.array(np.arange(0,8,2),dtype=str)
    # Plot the ticks
    for j in range(len(x_ticks)):
        ax.text(x_ticks[j],y_ticks[j],ticks[j])
    # Annotate the axes
    ax.text(0,-9,'Eastward distance (km)',fontsize=16,horizontalalignment='center',verticalalignment='center')
    ax.text(-9,0,'Northward distance (km)',fontsize=16,rotation=90,horizontalalignment='center',verticalalignment='center')
    # Make the grid
        # Lines
    angle = np.pi/6
    for i in range(6):
        angle = i*np.pi/6
        angle2 = angle+np.pi
        angles = np.array([angle,angle2])
        x = np.cos(angles)*8
        y = np.sin(angles)*8
        xtext = np.cos(angles)*8.5
        ytext = np.sin(angles)*8.5
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
    for i in range(5):
        circle = plt.Circle((0,0),i*2,fill=False,color='grey',alpha=0.4)
        ax.add_artist(circle)
    circle = plt.Circle((0,0),8,fill=False,color='k',alpha=1)
    ax.add_artist(circle)


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
    if orientation == 'EW' and eddy == 'A':
        # Anticylonic case and EW section
        y_max = np.nanmax(y)
        index = np.where(y == y_max)[0]
        if len(index)>1:
            index = index[0]
        x_center = x[index]
        y_center = y_max
    elif orientation == 'SN' and eddy == 'A':
        x_max = np.nanmax(x)
        index = np.where(x == x_max)[0]
        if len(index)>1:
            index = index[0]
        y_center = y[index]
        x_center = x_max
    # Plot the virtual center point
    center, = ax.plot(x_center,y_center,'k*',ms=10,label='Virtual center')
    # Legend
    ax.legend(handles = [first_point,last_point,center],bbox_to_anchor=(1.1, 1))

    return(fig)
