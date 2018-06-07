#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Functions to help with file reading, formatting and saving

import numpy as np
import datetime as dt

def FORSA_translate(filepath):
    """
    This functions opens the text file given by FORSA and makes a dictionnary
    nicely formatted
    """
    ## Open the file and make a nice array
    datafile = open(filepath)
    lines = datafile.readlines()
    temp = []
    for i in range(len(lines)):
        line = lines[i].strip().split('\t')
        temp.append(line)
    ## Making the dictionnary keys
    file_keys = np.array(temp[0])
    depth_idx = np.where(file_keys == 'cota') # The depths are always the same
    veloc_idx = np.where(file_keys == 'veloc')
    direc_idx = np.where(file_keys == 'direc')
    depths = np.array(np.array(temp[1])[depth_idx],dtype=float)
    ADCP_F = dict()
    longitudes = []
    latitudes = []
    directions = []
    velocities = []
    dates = []
    dnum = []
    Us = []
    Vs = []
    ## Extracting the data
    for i in range(1,len(temp)):
        veloc = np.array(np.array(temp[i])[veloc_idx],dtype=float)
        direc = np.array(np.array(temp[i])[direc_idx],dtype=float)
        direc_rad = np.deg2rad(direc + 90) # repère trigo 0 à droite
        U = veloc * np.cos(direc_rad)
        V = veloc * np.sin(direc_rad)
        longitude = np.float(temp[i][2])
        latitude = np.float(temp[i][1])
        date_hour_string = temp[i][0]
        date = dt.datetime.strptime(temp[i][0],'%d/%m/%Y %H:%M:%S')
        datenum = dt.datetime.toordinal(date) #this is just the day, not the hour
        longitudes.append(longitude)
        latitudes.append(latitude)
        dates.append(date)
        dnum.append(datenum)
        velocities.append(veloc)
        directions.append(direc)
        Us.append(U)
        Vs.append(V)
    ADCP_F['depths'] = depths
    ADCP_F['longitudes'] = np.array(longitudes)
    ADCP_F['latitudes'] = np.array(latitudes)
    ADCP_F['dates'] = np.array(dates)
    ADCP_F['datenum'] = np.array(dnum)
    ADCP_F['directions'] = np.array(directions)
    ADCP_F['velocities'] = np.array(velocities)
    ADCP_F['U'] = np.array(Us)
    ADCP_F['V'] = np.array(Vs)
    return(ADCP_F)

def Meteor_translate(filepath):
    """
    same thing as for FORSA but for the txt file produced with matlab from the Meteor data
    """
    #Opening
    datafile = open(filepath)
    lines = datafile.readlines()
    temp = []
    for i in range(len(lines)):
        line = lines[i].strip().split('\t')
        temp.append(line)
    #Making dict
    file_keys = np.array(temp[0])
    depth_idx = np.where(file_keys == 'depth') # The depths are always the same
    u_idx = np.where(file_keys == 'U')
    v_idx = np.where(file_keys == 'V')
    depths = np.array(np.array(temp[1])[depth_idx],dtype=float)
    ADCP = dict()
    longitudes = []
    latitudes = []
    directions = []
    velocities = []
    dates = []
    dnum = []
    Us = []
    Vs = []
    for i in range(1,len(temp)):
        u = np.array(np.array(temp[i])[u_idx],dtype=float)
        v = np.array(np.array(temp[i])[v_idx],dtype=float)
        direc = np.angle(u+v*1j,deg=True)
        veloc = np.sqrt(u**2 + v**2)
        longitude = np.float(temp[i][1])
        latitude = np.float(temp[i][2])
        time = temp[i][0]
        date = ref + dt.timedelta(time)
        ## transform the date from ugly matlab (? with unknown reference) to python
        longitudes.append(longitude)
        latitudes.append(latitude)
        dates.append(date)
        #     dnum.append(datenum)
        velocities.append(veloc)
        directions.append(direc)
        Us.append(u)
        Vs.append(v)
        nan_idx = np.where(np.isnan(longitudes))[0]
        longitudes = np.delete(longitudes,nan_idx)
        latitudes = np.delete(latitudes,nan_idx)
        dates = np.delete(dates,nan_idx)
        directions = np.delete(directions,nan_idx,0)
        velocities = np.delete(velocities,nan_idx,0)
        Us = np.delete(Us,nan_idx,0)
        Vs = np.delete(Vs,nan_idx,0)
    ADCP['depths'] = depths
    ADCP['longitudes'] = np.array(longitudes)
    ADCP['latitudes'] = np.array(latitudes)
    ADCP['dates'] = np.array(dates)
    #ADCP_M38['datenum'] = np.array(dnum)
    ADCP['directions'] = np.array(directions)
    ADCP['velocities'] = np.array(velocities)
    ADCP['U'] = np.array(Us)
    ADCP['V'] = np.array(Vs)
    return(ADCP)
