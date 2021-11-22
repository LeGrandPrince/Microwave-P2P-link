import numpy as np
import math
import pandas as pd

def powerThreshold(F_rx, B, M_QAM = "16-QAM", B_unit = "MHz"):
    '''
    Returns power treshold for specific QAM modulation.\n

    F_rx: Noise number.\n
    B: Bandwidth.\n
    M_QAM: String. Type of modulation. Default value: "16-QAM".\n
    B_unit: String. Units in which the bandwidth is given. 
    Since this is a microwave link script, default value is "MHz". 
    Ranges from "Hz" to "THz".
    
    
    '''
    #Dict with stored values of bandwidth frequencies
    B_units = {   'Hz':1,
                  'kHz':1*10**3,
                  'MHz':1*10**6,
                  'GHz':1*10**9,
                  'THz':1*10**12}

    #Calculate Thermal noise assuming noise density is a constant: 173.9 dBm/Hz.
    #Constant assumes that influence of value of above-the-sea level is insignificant in the expression.
    Pn = -173.9 + 10*np.log10(B*B_units[B_unit])  

    #BER-6 SNR thresholds for M-QAM modulations
    SNR_e6 = {'2-QAM':10.53,
              '4-QAM':13.2,
              '8-QAM':16.90,
              '16-QAM':20.5,
              '32-QAM':23.4,
              '64-QAM':26.6,
              '128-QAM':29.5,
              '256-QAM':32.6}

    #Power threshold for BER-6
    P_the6 = Pn + F_rx + SNR_e6[M_QAM]
    
    #Print the message
    print("Method powerThreshold has been called.\nPower threshold for " + M_QAM + " is " + str(np.round(P_the6,2)) + " dBm.\n\n\n")
    return P_the6

def flatFadeMargin(Ga, d, Ptx, fc, Pthe6):
    '''
    Calculates FFM for each link using given arguments. Function assumes using same antenna on every link.

    \nGa:antenna gain. Probably of type double.
    \nd: array of distances for each link.
    \nPtx: Transmitted power. Single value. Could be int, double or float.
    \nfc: Noise number. Assuming ideal transmitter it is 0. Usually int but can be double or float.
    \nPthe6: Power threshold for BER-6. Calculate using powerThreshold.

    '''

    #Create an array of length d full of Ptx and float 64 type
    Pt = np.full(len(d), Ptx, dtype = np.float64)
    
    #Calculation of effective isotropic radiated power
    P_eirp = Pt + Ga
    
    #Calculate link fade
    A_los = 20*np.log10(d)+20*np.log(fc) + 92.4
    
    #Calculate power on the reciever antenna
    P_rx = P_eirp - A_los + Ga

    #Calculate flat fading margin for BER-6
    FFM_e6 = P_rx - Pthe6
    print("Method flatFadeMargin has been called.\nFor transmitted power Ptn =  " + str(np.round(Pt, 2))+ " dBm, \nFFM = " + str(np.round(FFM_e6, 2)) + 
          ' dB.\nValues are rounded in the text window to 2 decimals, but are stored as np.float64.\n\n\n')

    #Calculate FFM/km
    FFM_km = FFM_e6 / d
    
    return FFM_e6

def setMinimumFFM(d):
    '''
    Function returns one value or array containing minimum FFM 
    reserves for each link. 
    \nd: value or array, distance.

    '''
    
    #Create dict with keys as upper distance boundaries and values as minimum FFM reserves.
    r = {#0-10km
         10: 17.5,
         #10-20km
         20: 22.5,
         #20-30km
         30: 27.5,
         #30-40km
         40: 32.5,
         #40-50km
         50: 37.5}

    #For purpose of later appending, create array full of zeros of length r
    max_distance_upper = np.zeros(len(r))
    
    #Append each key of dict r to max_distance_upper
    i = 0
    for v in r.keys():
        
        max_distance_upper[i] = v
        i = i+1

    #Set the lower distance boundaries
    max_distance_lower = max_distance_upper - 10

    #Create array full of zeros and length d for later appending
    min_FFM = np.zeros(len(d))
    
    #Filter values to find index of key for which min FFM is required
    i = 0
    for v in d:

        #Create arrays of booleans stored as ints for two below conditions.
        condition1 = (d[i] > max_distance_lower).astype(int)
        condition2 = (d[i] < max_distance_upper).astype(int)
        
        #Perform AND logical operation with 2 conditions.
        condition = condition1&condition2

        #Find index of value which is equal to 1
        index = [i for i, x in enumerate(condition) if x == 1]
        
        #Store the index as int and not array of one value
        index = index[0]
        
        #Change by the iteration value in min_FFM with value from dict r with key index calculated in the iteration
        min_FFM[i] = r[max_distance_upper[index].astype(int)]
        i=i+1
        

    print("Method setMinimumFFM has been called.\nMinimum FFM values for each link are: " 
          + str(min_FFM.round(2)) + " dBm.\n\n\n")
    return min_FFM
          
def transmittedPowerCorrection(Ptx, FFM, d):
    '''
    Corrects the value of transmitted Power for each link based on given distance.

    \nPtx: single value, could be int, double or float
    \nFFM: calculated values FFM from method flatFadeMargin
    \n d: array of distances for each link.

    '''

    #Create an np array of length d and values of Pt at each places
    Pt = np.full(len(d), Ptx, dtype = np.float64)

    #Create an array of booleans for given condition
    Ptx_bad = FFM < min

    #Find indices of values which satisfy the given condition
    indices = [i for i, x in enumerate(Ptx_bad) if x]

    #For each index which satisfies given condition, update the value by given formula
    for i in indices:
        Pt[i] = Pt[i] + (min[i] - FFM[i])
    
    #Print the results
    print('Method transmittedPowerCorrection has been called.\nCorrected values of Pt to satisfy the minimal FFM condition:\nPt = ' + 
          str(np.round(Pt, 2)) + 
          '\nValues in text are rounded to 2 decimals, but are stored as np.float64. \nCalculate FFM again with corrected values of Pt.\n\n\n')
    return Pt

#Specify noise number [dB]
F_rx = 3

#Specify bandwidth. Units are specified as function arguments.
B = 28

#Specify antennas gains [dBi]. If you use the same antenna for every link, specify only one. 
G = 38.9

#Distance between two links [km]
d = np.array([13.07, 26.52, 14.69, 16.95, 13.92, 18.86, 20.46, 7.24])

#Specify transmitted power [dBm]
Pt = 20

#Frequency [GHz]
Carrier_freq = 4

Link_names = ["ŽnH-Kunešov", "Kunešov-Tlstá", "Tlstá-Katova skala", "Katova skala-Rakytie", "Rakytie-Terchová", "Terchová-Kubínska", "Kubínska-Javorový vrch", "Javorový vrch-Trstenná"]

#Set minimum reserve for FFM
min = setMinimumFFM(d)

#Calculate the power threshold
Pte6 = powerThreshold(F_rx, B, "128-QAM", "MHz")

#Calculate FFM
FFM = flatFadeMargin(G, d, Pt, Carrier_freq,  Pte6)

#Perform transitted power correction to satisfy the minimum FFM values.
Pt = transmittedPowerCorrection(Pt, FFM, d)

#Calculate FFM for corrected values of Pt
FFM = flatFadeMargin(G, d, Pt, Carrier_freq,  Pte6)




