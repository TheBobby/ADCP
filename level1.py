#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Antonin Affholder'

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
    temp = np.array(temp)

    longitudes = np.array(temp[1:,2],dtype='float')
    latitudes = np.array(temp[1:,1],dtype='float')
    depths = np.array(temp[1,temp[0,:] == 'cota'],dtype='float')
    velocities = np.array(temp[1:,temp[0,:] == 'veloc'],dtype='float')
    directions = np.array(temp[1:,temp[0,:] == 'direc'],dtype='float')
    datestr = np.array(temp[1:,0],dtype='str')
    dates = np.array([dt.datetime.strptime(datestr[i],'%d/%m/%Y %H:%M:%S')
                      for i in range(len(datestr))])
    U = velocities * np.cos(np.deg2rad(directions + 90))
    V = velocities * np.sin(np.deg2rad(directions + 90))

    ADCP_F = dict()
    ADCP_F['depths'] = depths
    ADCP_F['longitudes'] = longitudes
    ADCP_F['latitudes'] = latitudes
    ADCP_F['dates'] = dates
    ADCP_F['directions'] = directions
    ADCP_F['velocities'] = velocities
    ADCP_F['U'] = U
    ADCP_F['V'] = V
    return(ADCP_F)

def Meteor_translate(filepath):
    """
    same thing as for FORSA but for the txt file produced with matlab from the Meteor data
    """
    #Opening
    datafile = open(filepath)
    lines = datafile.readlines()
    temp = []
    ref_date = dt.datetime(2016,1,1,0,0,0)
    for i in range(len(lines)):
        line = lines[i].strip().split('\t')
        temp.append(line)
    temp = np.array(temp)
    #Making dict
    depths = np.array(temp[1,temp[0,:] == 'depth'],dtype='float')
    longitudes = np.array(temp[1:,1],dtype='float') #because longitudes is on index 1
    latitudes = np.array(temp[1:,2],dtype='float')
    datenum = np.array(temp[1:,0],dtype='float')
    dates = np.array([ref_date + dt.timedelta(datenum[i])
                      for i in range(len(datenum))])
    U = np.array(temp[1:,temp[0,:] == 'U'],dtype='float')
    V = np.array(temp[1:,temp[0,:] == 'V'],dtype='float')
    velocities = np.sqrt(U**2 + V**2)
    directions = np.angle(U+V*1j,deg=True)
    ADCP = dict()

    ADCP['depths'] = depths
    ADCP['longitudes'] = longitudes
    ADCP['latitudes'] = latitudes
    ADCP['dates'] = dates
    ADCP['directions'] = directions
    ADCP['velocities'] = velocities
    ADCP['U'] = U
    ADCP['V'] = V
    return(ADCP)
