import csv
from math import sqrt
import math
from math import cos
from scipy import stats
from math import atan2, degrees

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d
import statistics
# import sklearn
import pandas as pd
from scipy.stats import circmean

from collections import defaultdict
#User guide
#when opening, look for long(short indicate section division) #####################################################################################################
#these indicate where the number of frames are divided to get time on the Y axis.
#replace if with whatever the fps are for the current video
#To change what file is assessed, head towards the bottom (on the final loop), below def main(): (see below)
#def main():
    # files = ["40dpiWT1CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv"]
    #Change the name here

def parseCoordsFromCSV(fileName):
    limbs = [
        "Snout",
        "LeftForelimb",
        "RightForelimb",
        "TopBody",
        "LeftHindlimb",
        "RightHindlimb",
        "BottomBody",
        "Tail2Base",
        "Tail15",
        "Tail1",
        "Tail05",
        "Tail0",
    ]

    limbCoordinates = {}
    for limb in limbs:
        limbCoordinates[limb] = []

    with open(fileName) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        print(fileName)
        # skip header rows
        for i in range(4):
            next(reader)

        # parse coordinates from csv file
        for row in reader:

            SnoutX = float(row[1]) if row[1] else None
            SnoutY = float(row[2]) if row[2] else None
            if SnoutX and SnoutY:
                limbCoordinates['Snout'].append((SnoutX, SnoutY))
            else:
                limbCoordinates['Snout'].append(None)

            LeftForelimbX = float(row[4]) if row[4] else None
            LeftForelimbY = float(row[5]) if row[5] else None
            if LeftForelimbX and LeftForelimbY:
                limbCoordinates['LeftForelimb'].append((LeftForelimbX, LeftForelimbY))
            else:
                limbCoordinates['LeftForelimb'].append(None)

            RightForelimbX = float(row[7]) if row[7] else None
            RightForelimbY = float(row[8]) if row[8] else None
            if RightForelimbX and RightForelimbY:
                limbCoordinates['RightForelimb'].append((RightForelimbX, RightForelimbY))
            else:
                limbCoordinates['RightForelimb'].append(None)

            TopBodyX = float(row[10]) if row[10] else None
            TopBodyY = float(row[11]) if row[11] else None
            if TopBodyX and TopBodyY:
                limbCoordinates['TopBody'].append((TopBodyX, TopBodyY))
            else:
                limbCoordinates['TopBody'].append(None)

            LeftHindlimbX = float(row[13]) if row[13] else None
            LeftHindlimbY = float(row[14]) if row[14] else None
            if LeftHindlimbX and LeftHindlimbY:
                limbCoordinates['LeftHindlimb'].append((LeftHindlimbX, LeftHindlimbY))
            else:
                limbCoordinates['LeftHindlimb'].append(None)

            RightHindlimbX = float(row[16]) if row[16] else None
            RightHindlimbY = float(row[17]) if row[17] else None
            if RightHindlimbX and RightHindlimbY:
                limbCoordinates['RightHindlimb'].append((RightHindlimbX, RightHindlimbY))
            else:
                limbCoordinates['RightHindlimb'].append(None)

            BottomBodyX = float(row[19]) if row[19] else None
            BottomBodyY = float(row[20]) if row[20] else None
            if BottomBodyX and BottomBodyY:
                limbCoordinates['BottomBody'].append((BottomBodyX, BottomBodyY))
            else:
                limbCoordinates['BottomBody'].append(None)

            Tail2BaseX = float(row[22]) if row[22] else None
            Tail2BaseY = float(row[23]) if row[23] else None
            if Tail2BaseX and Tail2BaseY:
                limbCoordinates['Tail2Base'].append((Tail2BaseX, Tail2BaseY))
            else:
                limbCoordinates['Tail2Base'].append(None)

            Tail15X = float(row[25]) if row[25] else None
            Tail15Y = float(row[26]) if row[26] else None
            if Tail15X and Tail15Y:
                limbCoordinates['Tail15'].append((Tail15X, Tail15Y))
            else:
                limbCoordinates['Tail15'].append(None)

            Tail1X = float(row[28]) if row[28] else None
            Tail1Y = float(row[29]) if row[29] else None
            if Tail1X and Tail1Y:
                limbCoordinates['Tail1'].append((Tail1X, Tail1Y))
            else:
                limbCoordinates['Tail1'].append(None)

            Tail05X = float(row[31]) if row[31] else None
            Tail05Y = float(row[32]) if row[32] else None
            if Tail05X and Tail05Y:
                limbCoordinates['Tail05'].append((Tail05X, Tail05Y))
            else:
                limbCoordinates['Tail05'].append(None)

            Tail0X = float(row[34]) if row[34] else None
            Tail0Y = float(row[35]) if row[35] else None
            if Tail0X and Tail0Y:
                limbCoordinates['Tail0'].append((Tail0X, Tail0Y))
            else:
                limbCoordinates['Tail0'].append(None)

        return limbCoordinates
#############################################################################################################
#calculate the total size of the mouse to get an understanding of what is too far for outlier
def length(limbCoordinates):
    Mouselength = []
    for idx in range(len(limbCoordinates["Snout"])):
        length = [
            limbCoordinates["Snout"][idx],
            limbCoordinates["Tail2Base"][idx],
        ]
        # filter out any positions which are undefined
        if length[0] and length[1]:
            # append coordinates
            x1 = length[0][0]
            y1 = length[0][1]

            x2 = length[1][0]
            y2 = length[1][1]

            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Mouselength.append(distance)
        else:
            Mouselength.append(None)
    Mouselength = [x for x in Mouselength if x is not None]
    return Mouselength
##############################################################################################################
#remove outlier values and replace with Nones
def findOutliers(limbCoordinates):
    limb_map = {
        'Snout': 0,
        'LeftForelimb': 1,
        'RightForelimb': 2,
        'TopBody': 3,
        'LeftHindlimb': 4,
        'RightHindlimb': 5,
        'BottomBody': 6,
        'Tail2Base': 7,
        'Tail15': 8,
        'Tail1': 9,
        'Tail05': 10,
        'Tail0': 11,
    }

    for limb_name, index in limb_map.items():
        for i in range(1, len(limbCoordinates[limb_name])):
            coord_current = limbCoordinates[limb_name][i]
            coord_last = limbCoordinates[limb_name][i - 1]

            if coord_current is None:
                continue

            x1 = coord_current[0]
            y1 = coord_current[1]

            if coord_last is None:
                # Search backwards for the closest previous non-None coordinate
                for j in range(i-2, i-10, -1):
                    if limbCoordinates[limb_name][j] is not None:
                        x2 = limbCoordinates[limb_name][j][0]
                        y2 = limbCoordinates[limb_name][j][1]
                        break
                else:
                    continue  # continue if there are no non-None coordinates before coord_last
            else:
                x2 = coord_last[0]
                y2 = coord_last[1]

            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance > 70 and limb_name in ['Tail2Base', 'Tail15', 'Tail1', 'Tail05', 'Tail0']:
                limbCoordinates[limb_name][i] = None

            if distance > 350 and limb_name in ['Snout', 'LeftForelimb', 'RightForelimb', 'TopBody', 'LeftHindlimb', 'RightHindlimb', 'BottomBody']:
                limbCoordinates[limb_name][i] = None

    return limbCoordinates
##########################################################################
#create function for statistics
def descriptive_stats(lst):
    """
    This function takes a list of numbers and its name, and returns a dictionary with the mean, median, min, max, and standard deviation.
    """
    n = len(lst)
    mean = sum(lst) / n
    median = sorted(lst)[n // 2] if n % 2 == 1 else sum(sorted(lst)[n // 2 - 1:n // 2 + 1]) / 2
    min_val = min(lst)
    max_val = max(lst)
    variance = sum((x - mean) ** 2 for x in lst) / n
    stdev = math.sqrt(variance)

    return {'mean': mean, 'median': median, 'min': min_val, 'max': max_val, 'stdev': stdev}
#######################################################################################################
#create function for removing jump outliers and skewing min hindlimb values
def extractOutSmall(lst):
    cleanList = []
    for value in lst:
        if value > 45:
            cleanList.append(value)
    return cleanList
#gather the distance between all points of interest####################################################
#Leg spread
def Lefttorightfoot(Interp_Coordinates):
    manspreads = []
    LegSpread = []

    for idx in range(len(Interp_Coordinates["Snout"])):
        manspread = [
            Interp_Coordinates["LeftHindlimb"][idx],
            Interp_Coordinates["RightHindlimb"][idx],
        ]
        # filter out any positions which are undefined
        if manspread[0] and manspread[1]:
            # append coordinates
            x1 = manspread[0][0]
            y1 = manspread[0][1]

            x2 = manspread[1][0]
            y2 = manspread[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            manspreads.append(distance)
            LegSpread.append((time, distance))
        else:
            manspreads.append(None)

    # filter out any None values from manspreads
    manspreads = [x for x in manspreads if x is not None]
    # filter outliers that are too small for animal behavior (45 to Baseline and presymptomatic maybe???)
    cleanManspread = extractOutSmall(manspreads)
    # gather statistics for list
    stats = descriptive_stats(cleanManspread)
    return stats

    #for printing the spread and saving the figure
    # plt.plot(*zip(*LegSpread))
    # plt.savefig(f'{fileName} - Average Spread Distance Plot.png')
    # plt.clf()
    # getting the spread over time per second (or with the 30, every 30 frames)
    # plt.plot(*zip(*LegSpread[::30]))
    # plt.savefig(f'{fileName} - Spread Distance Plot.png')
    # plt.clf()
################################################################
#forelimb spread
def Lefttorightarms(Interp_Coordinates):
    armspreads = []
    ArmDistance = []
    for idx in range(len(Interp_Coordinates["Snout"])):
        armspread = [
            Interp_Coordinates["LeftForelimb"][idx],
            Interp_Coordinates["RightForelimb"][idx],
        ]
        # filter out any positions which are undefined
        if armspread[0] and armspread[1]:
            # append coordinates
            x1 = armspread[0][0]
            y1 = armspread[0][1]

            x2 = armspread[1][0]
            y2 = armspread[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            armspreads.append(distance)
            ArmDistance.append((time, distance))
        else:
            armspreads.append(None)
    # filter out any None values from manspreads
    armspreads = [x for x in armspreads if x is not None]
    stats = descriptive_stats(armspreads)
    return stats
################################################################
#left stride length
def Leftstridelength(Interp_Coordinates):
    leftarmspreads = []
    LeftSpread = []

 # distance from left paw and left foot
    for idx in range(len(Interp_Coordinates["Snout"])):
        leftarmspread = [
            Interp_Coordinates["LeftForelimb"][idx],
            Interp_Coordinates["LeftHindlimb"][idx],
        ]

        # filter out any positions which are undefined
        if leftarmspread[0] and leftarmspread[1]:
            # append coordinates
            x1 = leftarmspread[0][0]
            y1 = leftarmspread[0][1]

            x2 = leftarmspread[1][0]
            y2 = leftarmspread[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            leftarmspreads.append(distance)
            LeftSpread.append((time, distance))
        else:
            leftarmspreads.append(None)
    # filter out any None values from manspreads
    leftarmspreads = [x for x in leftarmspreads if x is not None]
    stats = descriptive_stats(leftarmspreads)
    return stats
#############################################################
# right stride length
def Rightstridelength(Interp_Coordinates):
    rightarmspreads = []
    RArmSprd = []

    for idx in range(len(Interp_Coordinates["Snout"])):
        rightarmspread = [
             Interp_Coordinates["RightForelimb"][idx],
             Interp_Coordinates["RightHindlimb"][idx],
         ]

         # filter out any positions which are undefined
        if rightarmspread[0] and rightarmspread[1]:
            # append coordinates
            x1 = rightarmspread[0][0]
            y1 = rightarmspread[0][1]

            x2 = rightarmspread[1][0]
            y2 = rightarmspread[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            rightarmspreads.append(distance)
            RArmSprd.append((time, distance))
        else:
            rightarmspreads.append(None)
    rightarmspreads = [x for x in rightarmspreads if x is not None]
    stats = descriptive_stats(rightarmspreads)
    return stats
##################################################################
#distance from left paw to top body as left paw drag
def Leftpawtotop(Interp_Coordinates):
    leftpawdrags = []
    LpawDrag = []

    for idx in range(len(Interp_Coordinates["Snout"])):
        leftpawdrag = [
             Interp_Coordinates["LeftForelimb"][idx],
             Interp_Coordinates["TopBody"][idx],
         ]

         # filter out any positions which are undefined
        if leftpawdrag[0] and leftpawdrag[1]:
            # append coordinates
            x1 = leftpawdrag[0][0]
            y1 = leftpawdrag[0][1]

            x2 = leftpawdrag[1][0]
            y2 = leftpawdrag[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            leftpawdrags.append(distance)
            LpawDrag.append((time, distance))
        else:
            leftpawdrags.append(None)
    # filter out any None values from leftpawdrags
    leftpawdrags = [x for x in leftpawdrags if x is not None]
    stats = descriptive_stats(leftpawdrags)
    return stats
#########################################################################
#distance from right paw to top body as left paw drag
def Rightpawtotop(Interp_Coordinates):
    rightpawdrags = []
    RpawDrag = []
    for idx in range(len(Interp_Coordinates["Snout"])):
        rightpawdrag = [
             Interp_Coordinates["RightForelimb"][idx],
             Interp_Coordinates["TopBody"][idx],
         ]

         # filter out any positions which are undefined
        if rightpawdrag[0] and rightpawdrag[1]:
            # append coordinates
            x1 = rightpawdrag[0][0]
            y1 = rightpawdrag[0][1]

            x2 = rightpawdrag[1][0]
            y2 = rightpawdrag[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            rightpawdrags.append(distance)
            RpawDrag.append((time, distance))
        else:
            rightpawdrags.append(None)
    # filter out any None values from rightpawdrags
    rightpawdrags = [x for x in rightpawdrags if x is not None]
    stats = descriptive_stats(rightpawdrags)
    return stats
#########################################################################
#distance from top body to bottom to determine extension of torso
def ToptoBottom(Interp_Coordinates):
    Bodyextensions = []
    hunch = []
    for idx in range(len(Interp_Coordinates["Snout"])):
        Bodyextension = [
             Interp_Coordinates["TopBody"][idx],
             Interp_Coordinates["BottomBody"][idx],
         ]

         # filter out any positions which are undefined
        if Bodyextension[0] and Bodyextension[1]:
            # append coordinates
            x1 = Bodyextension[0][0]
            y1 = Bodyextension[0][1]

            x2 = Bodyextension[1][0]
            y2 = Bodyextension[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Bodyextensions.append(distance)
            hunch.append((time, distance))
        else:
            Bodyextensions.append(None)
    # filter out any None values from Bodyextensions
    Bodyextensions = [x for x in Bodyextensions if x is not None]
    stats = descriptive_stats(Bodyextensions)
    return stats
###################################################################
#Left leg drag
def Leftfoottobase(Interp_Coordinates):
    Leftfootdrags = []
    LfootDrag = []

    for idx in range(len(Interp_Coordinates["Snout"])):
        Leftfootdrag = [
            Interp_Coordinates["LeftHindlimb"][idx],
            Interp_Coordinates["Tail2Base"][idx],
        ]

        # filter out any positions which are undefined
        if Leftfootdrag[0] and Leftfootdrag[1]:
            # append coordinates
            x1 = Leftfootdrag[0][0]
            y1 = Leftfootdrag[0][1]

            x2 = Leftfootdrag[1][0]
            y2 = Leftfootdrag[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Leftfootdrags.append(distance)
            LfootDrag.append((time, distance))
        else:
            Leftfootdrags.append(None)
    # filter out any None values from Leftfootdrags
    Leftfootdrags = [x for x in Leftfootdrags if x is not None]
    stats = descriptive_stats(Leftfootdrags)
    return stats
###################################################################
#Right leg drag
def Rightfoottobase(Interp_Coordinates):
    Rightfootdrags = []
    rate_of_changes = []
    RfootDrag = []

    for idx in range(len(Interp_Coordinates["Snout"])):

        Rightfootdrag = [
            Interp_Coordinates["RightHindlimb"][idx],
            Interp_Coordinates["Tail2Base"][idx],
        ]

        # filter out any positions which are undefined
        if Rightfootdrag[0] and Rightfootdrag[1]:
            # append coordinates
            x1 = Rightfootdrag[0][0]
            y1 = Rightfootdrag[0][1]

            x2 = Rightfootdrag[1][0]
            y2 = Rightfootdrag[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Rightfootdrags.append(distance)
            RfootDrag.append((time, distance))
        else:
            Rightfootdrags.append(None)
    # filter out any None values from Rightfootdrags
    Rightfootdrags = [x for x in Rightfootdrags if x is not None]
    stats = descriptive_stats(Rightfootdrags)
    return stats
#####################################################################
#Distance from tail base to tail1.5
def T2toT15(Interp_Coordinates):
    baseto15s = []
    Tail0Tuple = []
    for idx in range(len(Interp_Coordinates["Snout"])):

        baseto15 = [
            Interp_Coordinates["Tail2Base"][idx],
            Interp_Coordinates["Tail15"][idx],
        ]

        # filter out any positions which are undefined
        if baseto15[0] and baseto15[1]:
            # append coordinates
            x1 = baseto15[0][0]
            y1 = baseto15[0][1]

            x2 = baseto15[1][0]
            y2 = baseto15[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            baseto15s.append(distance)
            Tail0Tuple.append((time, distance))
        else:
            baseto15s.append(None)
    return baseto15s
###############################################################################
#Distance from tail1.5 to tail1
def tail15totail1(Interp_Coordinates):
    Tail15to1s = []
    Tail2Tuple = []

    for idx in range(len(Interp_Coordinates["Snout"])):

        Tail15to1 = [
            Interp_Coordinates["Tail15"][idx],
            Interp_Coordinates["Tail1"][idx],
        ]

        # filter out any positions which are undefined
        if Tail15to1[0] and Tail15to1[1]:
            # append coordinates
            x1 = Tail15to1[0][0]
            y1 = Tail15to1[0][1]

            x2 = Tail15to1[1][0]
            y2 = Tail15to1[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Tail15to1s.append(distance)
            Tail2Tuple.append((time, distance))
        else:
            Tail15to1s.append(None)
    return Tail15to1s
######################################################################################
#Distance from tail1 to tail0.5
def tail1totail05(Interp_Coordinates):
    Tail1to05s = []
    Tail3Tuple = []
    for idx in range(len(Interp_Coordinates["Snout"])):

        Tail1to05 = [
            Interp_Coordinates["Tail1"][idx],
            Interp_Coordinates["Tail05"][idx],
        ]

        # filter out any positions which are undefined
        if Tail1to05[0] and Tail1to05[1]:
            # append coordinates
            x1 = Tail1to05[0][0]
            y1 = Tail1to05[0][1]

            x2 = Tail1to05[1][0]
            y2 = Tail1to05[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Tail1to05s.append(distance)
            Tail3Tuple.append((time, distance))
        else:
            Tail1to05s.append(None)
    return Tail1to05s
##############################################################################################
#Distance from tail0.5 to tail0
def tail05totail0(Interp_Coordinates):
    Tail05to0s = []
    Tail4Tuple = []
    for idx in range(len(Interp_Coordinates["Snout"])):

        Tail05to0 = [
            Interp_Coordinates["Tail05"][idx],
            Interp_Coordinates["Tail0"][idx],
        ]

        # filter out any positions which are undefined
        if Tail05to0[0] and Tail05to0[1]:
            # append coordinates
            x1 = Tail05to0[0][0]
            y1 = Tail05to0[0][1]

            x2 = Tail05to0[1][0]
            y2 = Tail05to0[1][1]
            time = idx / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            Tail05to0s.append(distance)
            Tail4Tuple.append((time, distance))
        else:
            Tail05to0s.append(None)
    print(idx)
    return Tail05to0s
#################################################################################################
#Tracking of body based on Tailbase
def TailTracking(Interp_Coordinates):
    # coords (time, distance, velocity, acceleration)
    BaseTailList = []
    VelocityTailList = []
    AccelerationTail = []
    cumulativeDistance = 0
    prevTime = 0
    prevDist = 0
    prevVel = 0
    prevAcc = 0

    coord_list = [coord for coord in Interp_Coordinates["Tail2Base"]]
    SpeedList = []
    VelocityList = []
    Acceleration_list = []
    for i in range(len(coord_list) - 1):
        #this verifies that the current and next coordinates are not none, if there is a none, it skips to the next set to calculate distance
        if None in coord_list[i:i + 2]:
            continue

        x1 = coord_list[i][0]
        y1 = coord_list[i][1]

        x2 = coord_list[i + 1][0]
        y2 = coord_list[i + 1][1]
        time = i / 25
        # calculate distance and cumulative distance
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cumulativeDistance += distance
        BaseTailList.append((time, cumulativeDistance))
        # calculate speed and acceleration
        deltaTime = time - prevTime
        deltaDist = cumulativeDistance - prevDist
        # if the deltatime is 0, it means there is no movement and the speed is 0, therefore it just adds the speed to zero to avoid dividing my 0
        speed = deltaDist / deltaTime if deltaTime != 0 else 0
        SpeedList.append(speed)
        prevTime = time
        prevDist = cumulativeDistance
        prevVel = speed
        velocity = distance / deltaTime if deltaTime != 0 else 0
        VelocityTailList.append((time, velocity))
        VelocityList.append(velocity)
        acceleration = (velocity - prevVel) / deltaTime if deltaTime != 0 else 0
        Acceleration_list.append(acceleration)
        AccelerationTail.append((time, acceleration))
    Speed = cumulativeDistance / time
    # plot distance
    # plot distance
    plt.subplot(3, 1, 1)
    plt.plot([x[0] for i, x in enumerate(BaseTailList) if i % 25 == 0],
             [x[1] for i, x in enumerate(BaseTailList) if i % 25 == 0])
    plt.ylabel('Distance')

    # plot velocity
    plt.subplot(3, 1, 2)
    plt.plot([x[0] for i, x in enumerate(VelocityTailList) if i % 25 == 0],
             [x[1] for i, x in enumerate(VelocityTailList) if i % 25 == 0])
    plt.ylabel('Velocity')

    # plot acceleration
    plt.subplot(3, 1, 3)
    plt.plot([x[0] for i, x in enumerate(AccelerationTail) if i % 25 == 0],
             [x[1] for i, x in enumerate(AccelerationTail) if i % 25 == 0])
    plt.ylabel('Acceleration')
    # set the x-axis label for the last subplot
    plt.xlabel('Time in Sec')
    # save the figure
    # plt.savefig(f'{fileName} - Time Distance Plot.png')
    # show the plot
    # plt.show()
    print("Overall speed of TailBase", Speed)
    statsSpeed = descriptive_stats(SpeedList)
    # , 'TailBase Speed'
    descriptive_stats(VelocityList)
    # , 'TailBase Velocity'
    statsAcc = descriptive_stats(Acceleration_list)
    # , 'TailBase Acceleration'
    return statsSpeed, statsAcc
##########################################################################
#make equation for calculating distance, speed, velocity, acceleration
def MotionTrack(Coordinates):
    limbSpeedList = {}
    limbVelocityList = {}
    limbAccelerationList = {}
    cumDistDict = {}
    for parts in Coordinates:
        LimbList = [coord for coord in Coordinates[parts]]
        cumulativeDistance = 0
        # Travel = []
        prevTime = 0
        prevDist = 0
        prevVel = 0
        speedList = []
        velocityList = []
        accelerationList = []
        cumDistanceList = []
        for i in range(len(LimbList) - 1):
            # this verifies that the current and nest coordinates are not none, if there is a none, it skips to the next set to calculate
            if None in LimbList[i:i + 2]:
                continue
            x1 = LimbList[i][0]
            y1 = LimbList[i][1]

            x2 = LimbList[i + 1][0]
            y2 = LimbList[i + 1][1]
            time = i / 25
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            cumulativeDistance += distance
            # Travel.append((time, cumulativeDistance))
            cumDistanceList.append(cumulativeDistance)
            deltaTime = time - prevTime
            deltaDist = cumulativeDistance - prevDist
            speed = deltaDist / deltaTime if deltaTime != 0 else 0
            speedList.append(speed)
            prevTime = time
            prevDist = cumulativeDistance
            prevVel = speed
            velocity = distance / deltaTime if deltaTime != 0 else 0
            # velocityList.append((time, velocity))
            velocityList.append(velocity)
            acceleration = (velocity - prevVel) / deltaTime if deltaTime != 0 else 0
            accelerationList.append(acceleration)
        limbSpeedList[parts] = speedList
        limbVelocityList[parts] = velocityList
        limbAccelerationList[parts] = accelerationList
        cumDistDict[parts] = cumDistanceList
    print('Cummulative Distance', cumulativeDistance)
    return limbAccelerationList, limbSpeedList, limbVelocityList, cumDistDict

##########################################################################
#moving averages function: SMA simple moving average
def SMA(lst, windowSize):
    i = 0
    moving_averages = []
    while i < len(lst) - windowSize + 1:
        window = lst[i : i + windowSize]
        window_average = round(np.sum(window) / windowSize, 2)
        moving_averages.append(window_average)
        i += 1
    return moving_averages
##########################################################################
#getting the speed and acceleration for each limb
def limbMovement(Interp_Coordinates):
    #Acceleration used to graph oscillation pattern over time
    Acceleration, Speed, Velocity, CumulativeDistance = MotionTrack(Interp_Coordinates)
    time = [i / 25 for i in range(len(Interp_Coordinates["Snout"]))]
    LFa = Acceleration["LeftForelimb"]
    statsLFA = descriptive_stats(LFa)
    # , 'Left Forelimb Acceleration'
    RFa = Acceleration["RightForelimb"]
    statsRFA = descriptive_stats(RFa)
    # , 'Right Forelimb Acceleration'
    LHa = Acceleration["LeftHindlimb"]
    statsLHA = descriptive_stats(LHa)
    # , 'Left Hindlimb Acceleration'
    RHa = Acceleration["RightHindlimb"]
    statsRHA = descriptive_stats(RHa)
    # , 'Right Hindlimb Acceleration'
    #Speed
    LFS = Speed["LeftForelimb"]
    statsLFS = descriptive_stats(LFS)
    # , 'Left Forelimb Speed'
    RFS = Speed["RightForelimb"]
    statsRFS = descriptive_stats(RFS)
    # , 'Right Forelimb Speed'
    LHS = Speed["LeftHindlimb"]
    statsLHS = descriptive_stats(LHS)
    # , 'Left Hindlimb Speed'
    RHS = Speed["RightHindlimb"]
    statsRHS = descriptive_stats(RHS)
    # , 'Right Hindlimb Speed'
    #Velocity
    LFv = Velocity["LeftForelimb"]
    descriptive_stats(LFv)
    # , 'Left Forelimb Velocity'
    RFv = Velocity["RightForelimb"]
    descriptive_stats(RFv)
    # , 'Right Forelimb Velocity')
    LHv = Velocity["LeftHindlimb"]
    descriptive_stats(LHv)
    # , 'Left Hindlimb Velocity'
    RHv = Velocity["RightHindlimb"]
    descriptive_stats(RHv)
    # , 'Right Hindlimb Velocity'
    # # Distance
    # LFd = CumulativeDistance["LeftForelimb"]
    # RFd = CumulativeDistance["RightForelimb"]
    # LHd = CumulativeDistance["LeftHindlimb"]
    # RHd = CumulativeDistance["RightHindlimb"]

    # # Only plotting data for the first 30 seconds as otherwise it's too much
    # time = time[:750]
    # LF = LFa[:750]
    # RF = RFa[:750]
    # LH = LHa[:750]
    # RH = RHa[:750]
    # # Only plot data every 5 seconds
    # time = time[::5]
    # LF = LF[::5]
    # RF = RF[::5]
    # LH = LH[::5]
    # RH = RH[::5]
    # # plt.subplot(4, 1, 1)
    # plt.plot(time, LF, label='LeftForelimb')
    # #plt.subplot(4, 1, 2)
    # plt.plot(time, RF, label='RightForelimb')
    # # plt.subplot(4, 1, 3)
    # plt.plot(time, LH, label='LeftHindlimb')
    # # plt.subplot(4, 1, 4)
    # plt.plot(time, RH, label='RightHindlimb')
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Distance Traveled (pixels)')
    # plt.show()
    return statsLFA, statsRFA, statsLHA, statsRHA, statsLFS, statsRFS, statsLHS, statsRHS
##################################################################################################
#make negative coordinates so that y coordinates make sense
def makeNeg(Coordinates):
    Neg_Coordinates = {}
    for parts in Coordinates:
        NegY = []
        for coord in Coordinates[parts]:
            if coord is not None:
                x, y = coord
                NegY.append((x, -y))
            else:
                NegY.append(None)
        Neg_Coordinates[parts] = NegY

    return Neg_Coordinates
####################################################################################################################
#Angles of left hindlimb, tailbase, and right hindlimb
def ParalysisAngle(Interp_Coordinates):
    LHtoTBtoRH_angles = []
    Neg_Coordinates = makeNeg(Interp_Coordinates)
    for idx in range(len(Interp_Coordinates["Snout"])):
        LH = Neg_Coordinates["LeftHindlimb"][idx]
        TB = Neg_Coordinates["Tail2Base"][idx]
        RH = Neg_Coordinates["RightHindlimb"][idx]
        if LH and TB and RH:
            angle = []
            smallangle = []
            #this if statement makes sure there are at least 2 coordinates in the condition before it calculates the angle
            if len(LH) >= 2 and len(TB) >= 2 and len(RH) >= 2:
                # calculate angle between LH and RH with TB as vertex
                # theta_LH = math.atan2(LH[1]-TB[1], LH[0]-TB[0])
                # theta_RH = math.atan2(RH[1]-TB[1], RH[0]-TB[0])
                # angle = math.degrees(theta_RH - theta_LH) % 360
                deg1 = (360 + degrees(atan2(RH[0] - TB[0], RH[1] - TB[1]))) % 360
                deg2 = (360 + degrees(atan2(LH[0] - TB[0], LH[1] - TB[1]))) % 360
                angle = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)
                #this is to always get the internal angle
                if angle > 180:
                    angle = 360 - angle
                LHtoTBtoRH_angles.append(angle)
            else:
                LHtoTBtoRH_angles.append(None)
    stats = descriptive_stats(LHtoTBtoRH_angles)

    # creates circular graph to display data
    min_angle = np.min(LHtoTBtoRH_angles)
    max_angle = np.max(LHtoTBtoRH_angles)
    MedLHtoTBtoRH_angle = np.median(LHtoTBtoRH_angles)
    ax = plt.subplot(polar=True)
    ax.plot([0, np.radians(MedLHtoTBtoRH_angle)], [0, 1], color="darkorange", linewidth=2)
    ax.set_rticks([])
    ax.set_rlim([0, 1])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    #   #makes lines that target the max angles and min angles of the animal
    ax.plot([0, np.radians(min_angle)], [0, 1], "b")
    ax.plot([0, np.radians(max_angle)], [0, 1], "b")
    ax.fill_between(np.linspace(np.radians(max_angle), np.radians(min_angle), 100), 0, 1, alpha=0.4, color="dodgerblue")
    plt.show()
    return stats
##############################################################################################
#create function for removing tail outliers
def extractOut(lst):
    cleanList = []
    for value in lst:
        if value < 300:
            cleanList.append(value)
    return cleanList
##############################################################################################Final loop
#to itterate over the main file and all documents
def main():
    files = [
        "Baseline1CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline2CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline3CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline4CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline5CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT1CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT2CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT3CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT4CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT5CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT6CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv"
]
    #Baseline1CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline2CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline3CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline4CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv","Baseline5CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT1CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT2CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT3CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT4CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT5CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv", "10dpiWT6CropDLC_resnet152_152EAEWTMar2shuffle1_80000_sk_filtered.csv"
    mouse_dfs = []
    mouseDistanceMap = {file: 0 for file in files}
    for file in files:
        TotalTailLength = []
        limbCoordinates = parseCoordsFromCSV(file)
        #interpolate coordinates to remove outlier points that jump
        Interp_Coordinates = findOutliers(limbCoordinates)
        animalName = file.split("Crop")[0]
        #parsing into a dataframe
        MousepxRef = length(Interp_Coordinates)
        NegYCoords = makeNeg(Interp_Coordinates)
        DistanceTrav = MotionTrack(Interp_Coordinates)
        LFa, LRa, LHa, RHa, LFs, RFs, LHs, RHs = limbMovement(Interp_Coordinates)
        TailSpeed, TailAcc = TailTracking(Interp_Coordinates)
        Angle = ParalysisAngle(Interp_Coordinates)
        #Hindlimb Spread
        HindlimbSpread = Lefttorightfoot(Interp_Coordinates)
        # Forelimb Spread
        ForelimbSpread = Lefttorightarms(Interp_Coordinates)
        #Left side stride
        LStrides = Leftstridelength(Interp_Coordinates)
        # Right side stride
        RStrides = Rightstridelength(Interp_Coordinates)
        #Left paw drag
        LDrags = Leftpawtotop(Interp_Coordinates)
        # Right paw drag
        RDrags = Rightpawtotop(Interp_Coordinates)
        # Top to bottom extension of mouse torso
        BodyExtensions = ToptoBottom(Interp_Coordinates)
        # Left foot drag
        LfootDrags = Leftfoottobase(Interp_Coordinates)
        # Right foot drag
        RfootDrags = Rightfoottobase(Interp_Coordinates)
        # Distance from tail base to tail 1.5
        Tail2_15s = T2toT15(Interp_Coordinates)
        # Distance from tail 1.5 to tail 1
        Tail15_1s = tail15totail1(Interp_Coordinates)
        # Distance from tail 1 to tail 0.5
        Tail1_05s = tail1totail05(Interp_Coordinates)
        # Distance from tail 0.5 to tail 0
        Tail05_0s = tail05totail0(Interp_Coordinates)

        # Average the distances from the base to the tip of the tail and append to list
        LengthofTail = []
        if Tail2_15s is not None and Tail15_1s is not None and Tail1_05s is not None and Tail05_0s is not None:
            for k in zip(Tail2_15s, Tail15_1s, Tail1_05s, Tail05_0s):
                if all(elem is not None for elem in k):
                    LengthofTail.append(np.sum(k))
                else:
                    LengthofTail.append(None)
        else:
            TotalTailLength.append(None)
        TotalTailLength.append(LengthofTail)
        flattened_list = [item for sublist in TotalTailLength for item in sublist if item is not None]
        FinalTailLength = extractOut(flattened_list)
        TailLength = descriptive_stats(FinalTailLength)

        labels = [
            "Left Forelimb Acceleration", "Right Forelimb Acceleration", "Left Hindlimb Acceleration",
            "Right Hindlimb Acceleration", "Left Forelimb Speed", "Right Forelimb Speed", "Left Hindlimb Speed",
            "Right Hindlimb Speed", "TailBase Speed", "TailBase Acceleration",
            "Paralysis Angle","Hindlimb Spread", "Forelimb Spread", "Left Stride Length", "Right Stride Length",
            "Left Paw Drag","Right Paw Drag", "Body Extension", "Left foot Drag", "Right foot Drag", "Tail Length"
        ]

        #create first dataframe
        dataframes = []
        measures = [
            LFa,
            LRa,
            LHa,
            RHa,
            LFs,
            RFs,
            LHs,
            RHs,
            TailSpeed,
            TailAcc,
            Angle,
            HindlimbSpread,
            ForelimbSpread,
            LStrides,
            RStrides,
            LDrags,
            RDrags,
            BodyExtensions,
            LfootDrags,
            RfootDrags,
            TailLength,
        ]
        for i, measure in enumerate(measures):
            iterables = [[labels[i]], measure.keys()]
            index = pd.MultiIndex.from_product(iterables, names=["Phenotype", "Metric"])
            dataframe = pd.DataFrame(
                data=measure.values(),
                index=index,
                columns=[animalName]
            )
            # print(dataframe)
            dataframes.append(dataframe)

        stacked_df = pd.concat(dataframes)
        mouse_dfs.append(stacked_df)
        # stacked_df.to_excel("total_output.xlsx")
    final_df = pd.concat(mouse_dfs, axis=1)
    final_df.to_excel("Final_Output.xlsx")
    # print(final_df)

if __name__ == '__main__':
    main()

# how to write a list to excel
    # df = pd.DataFrame("your list name")
    # writer = pd.ExcelWriter('Name.xlsx', engine='xlsxwriter')
    # df.to_excel(writer, sheet_name='Distance', index=True)
    # writer.save()