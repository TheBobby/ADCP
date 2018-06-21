#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Antonin Affholder'

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

def InsidePolygon(X,Y,x,y):
    '''
            *** Function InsidePolygon ***
    Finds points in the plan that are inside the surface defined by a given
    polygon
            *** Arguments ***
    - X,Y are the coordinates of the contour points
    - x,y are the coordinates of the point to test
            *** Outputs ***
    Returns True if x,y is inside the contour X,Y; False otherwise
            *** Remarks ***
    Kind of useless
    '''
    ## First make a path object from the contour coordinates
    vertices = [(X[i],Y[i]) for i in range(len(X))]
    vertices.append((X[0],Y[0]))
    path = pth.Path(vertices)
    inside = path.contains_point([x,y])
    return(inside)

def PolygonSubsetter(data,Xc,Yc):
    '''
            *** Funciton PolygonSubsetter ***
    Retrieves the data that is inside a region defined by a polygon
            *** Arguments ***
    - data is ADCP data well formatted
    - Xc are the x coordinates of the polygon
    - Yc are the y coordinates of the polygon
            *** Outputs ***
    - subadcp is a dict object, same formatting as original data as given by
    file_ADCP functions
            *** Remarks ***
    IT WILL NOT SUBSET LVL 2 INFO SUCH AS VORTICITY
    Is aimed to be used to easily extract data inside a structure defined by a
    contour
    '''
    vertices = [(Xc[i],Yc[i]) for i in range(len(Xc))]
    vertices.append((Xc[0],Yc[0]))
    path = pth.Path(vertices)
    indexes = []
    for i in range(len(data['longitudes'])):
        x = data['longitudes'][i]
        y = data['latitudes'][i]
        val = path.contains_point([x,y])
        indexes.append(val)
    indexes = np.array(indexes)
    lons = data['longitudes'][indexes]
    lats = data['latitudes'][indexes]
    U = data['U'][indexes,:]
    V = data['V'][indexes,:]
    direct = data['directions'][indexes,]
    veloc = data['velocities'][indexes,]
    dates = data['dates'][indexes]
    subadcp = {'longitudes':lons,'latitudes':lats,'dates':dates,
               'depths':data['depths'],
               'U':U,'V':V,
               'directions':direct,'velocities':veloc}
    return(subadcp)

def BoxSubsetter(box,data):
    """
    OUTDATED BY PolygonSubsetter
    Subsets the ADCP data inside a rectangular box
    data is ADCP data formatted correctly, see previous functions and
    introduction
    """
    ## First the regular data
    lon = data['longitudes']
    lat = data['latitudes']
    lonmin = box[0][0]
    lonmax = box[0][1]
    latmin = box[1][0]
    latmax = box[1][1]
    lonmin_bool = lon <= lonmax
    lonmax_bool = lon >= lonmin
    latmin_bool = lat <= latmax
    latmax_bool = lat >= latmin
    lon_idx = np.where(lonmax_bool == lonmin_bool)[0]
    lat_idx = np.where(latmax_bool == latmin_bool)[0]
    minlen = min(len(lon_idx),len(lat_idx))
    if len(lon_idx) == minlen:
        idx = lon_idx
    elif len(lat_idx) == minlen:
        idx = lat_idx
    sublon = lon[idx]
    sublat = lat[idx]
    U = data['U'][idx,:]
    V = data['V'][idx,:]
    subdata = {'longitudes':sublon,'latitudes':sublat,
               'U':U,'V':V,
               'depths':data['depths']}
    return(subdata)

### Manipulate the coordinates
# Here are funcitons that allow different kind of projections and data cleaning
# This will provide ~level 3 data

def OrthogoalProjection(a,b,x,y):
    """
            *** Function OrthogoalProjection ***
    Returns the coordinates of the points projected orthogonally on a line of
    equation a + b*x
            *** Arguments ***
    - a,b are the coefficients defining the line a + b*x
    - x,y are the coordinates of the points to project (can be np arrays)
    *** Outputs ***
    - projx,projy are the coordinates in the plan of the orthogonally projected
    points
            *** Remarks ***
    """
    # lon and lat in projected on the map coordinates
    # a + b*x
    projx = (x + b*(y-a))/(1 + b**2)
    projy = projx * b + a
    return(projx,projy)

def Distances(Xs,Ys):
    '''
    computes the distances between the points
    USELESS
    '''
    d = np.sqrt((Xs - Xs[0])**2 + (Ys - Ys[0])**2)
    return(d)

def Rotation(xs,ys,theta,origin):
    '''
            *** Function Rotation ***
    Gives the coordinates in a new frame given by new origin and a rotation angle
            *** Arguments ***
    - theta is the angle of rotation between the two frames. Taken positive
    counterclockwise
    - origin are the coordinates (tuple object) of the new origin in the old frame
            *** Outputs ***
    - X,Y are the coordinates of the points xs,ys in the new frame
    '''
    Ro = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    X = []
    Y = []
    O = np.matrix([[origin[0]],[origin[1]]])
    for i in range(len(xs)):
        xi = xs[i]
        yi = ys[i]
        coords = np.matrix([[xi],[yi]])
        ncoords = Ro.I * (coords - O)
        Xi = ncoords[0,0]
        Yi = ncoords[1,0]
        X.append(Xi)
        Y.append(Yi)
    X = np.array(X)
    Y = np.array(Y)
    return(X,Y)

def EddyFrame(data_sub,angle=0):
    """
            *** EddyFrame Function ***
    Makes a compact dataset of what is useful for an Eddy in ADCP in a frame that
    is nice -which is a nice line that crosses the center of the eddy and the
    dataset is nicely in order
            *** Arguments ***
    - data_sub is the dictionnary of the already subsetted ADCP data, as obtained
    with PolygonSubsetter
    - Xcenter is the longitude of the center of the Eddy
    - Ycenter is the latitude of the center of the Eddy
    - b is the factor of rotation of the new frame, default is 0 for longitudinal line
    but for NS line, it should be np.pi/2
            *** Outputs ***
    - x is the x values in a new frame that takes for origin the center of the eddy
    and that is a line from West to East
    - v is the velocity perpendicular to this new line (up = positive), in this case
    where the line is perfectly zonal, v is the meridional velocity but this function
    is easy to change to new frames
    - z is the depths
    """
    ADCP = data_sub
    # Get the coordinates in spherical
    lon = ADCP['longitudes']
    lat = ADCP['latitudes']
    # Get the velocities
    U = ADCP['U']
    V = ADCP['V']
    # Set up the projection (mercator)
    m = Basemap(projection='merc',
                llcrnrlat=min(lat),urcrnrlat=max(lat),
                llcrnrlon=min(lon),urcrnrlon=max(lon),
                resolution='c')
    # Project the stations coordinates
    x,y = m(lon,lat)
    ## Find the center
    # Project on a horizontal line
    mU = np.nanmean(U[:,:10],axis=1)
    mV = np.nanmean(V[:,:10],axis=1)
    U_temp,V_temp = Rotation(mU,mV,angle,[0,0]) #for the vectors, the origins do not change, only their orientation
    # Find the center
    # V_abs = np.abs(nV)
    indexes = np.where(np.isfinite(V_temp))[0]
    V_temp = V_temp[indexes]
    U_temp = U_temp[indexes]
    nx = x[indexes]
    ny = y[indexes]
    xcoeffs = np.polyfit(nx,V_temp,1)
    xline = np.polyval(xcoeffs,nx)
    xline_abs = np.abs(xline)
    x_center = nx[np.where(xline_abs==min(xline_abs))][0]
    y_center = np.mean(ny)
    # Coefficients of a straight longitudinal line going through the center
    a = y_center

    # Getting the parameters of the new frame
    origin = [x_center,y_center] #origin

    # Projecting everything in the new frame
    nU = []
    nV = []
    for i in range(U.shape[1]):
        us = U[:,i]
        vs = V[:,i]
        nus,nvs = Rotation(us,vs,angle,[0,0]) #for the vectors, the origins do not change, only their orientation
        nU.append(nus)
        nV.append(nvs)
    nU = np.transpose(np.array(nU))
    nV = np.transpose(np.array(nV))
    #Project the stations coordinates
    nX,nY = Rotation(x,y,angle,origin)
    # Put everything in a nice order and managing doubles
    nXs = list(set(nX))
    nXs.sort()
    nVs = []
    # Managing doubles
    for x in nXs:
        idx = np.where(nX == x)[0]
        Vs = np.nanmean(nV[idx,:],axis=0)
        nVs.append(Vs)
    nVs = np.array(nVs)
    X_center,Y_center = m(x_center,y_center,inverse=True)
    return(nXs,ADCP['depths'],nVs,X_center,Y_center)



# JIC, the coordinates of the projected points in the original frame
# px,py = OrthogoalProjection(a,b,x,y)
## If instead of choosing a longitudinal line, we want to use a linear
## regression of the stations coordinates; remember to uncomment the import
## of the sklearn package!!
# Projecting on a regression line
# regr = linear_model.LinearRegression()
# regr.fit(x.reshape(len(x),1), y.reshape(len(y),1))
# b = regr.coef_[0,0]
# a = regr.predict(0)[0,0]
# px,py = OrthogoalProjection(a,b,x,y)
