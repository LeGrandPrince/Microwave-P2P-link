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

    \nGa:antenna gain. Usually double [dBi] but can be int as well.
    \nd: array of distances for each link [km]. Usually int or double.
    \nPtx: Transmitted power [dBm]. Single value. Could be int, double or float.
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
         10: 15,
         #10-20km
         20: 20,
         #20-30km
         30: 25,
         #30-40km
         40: 30,
         #40-50km
         50: 35}

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
          
def transmittedPowerCorrection(Ptx, FFM, d, min):
    '''
    Corrects the value of transmitted Power for each link based on given distance.

    \nPtx: single value, could be int, double or float
    \nFFM: calculated values FFM from method flatFadeMargin
    \n d: array of distances for each link.

    '''

    #Create an np array of length d and values of Pt at each places
    Pt = np.full(len(d), Ptx, dtype = np.float64)

    #Create an array of booleans for FFM is smaller than minimal FFM value (array of True/False values)
    Ptx_bad_min = FFM < min
    Ptx_bad_max = FFM > min + 5
    
    #Perform OR logical operation on given values and store it as Ptx_bad
    Ptx_bad = Ptx_bad_min|Ptx_bad_max

    #Find indices of values which satisfy the given condition
    indices = [i for i, x in enumerate(Ptx_bad) if x]

    #For each index which satisfies given condition, update transmitted power Pt by given formula
    for i in indices:
        print(i)
        Pt[i] = Pt[i] + (min[i] - FFM[i])
    
    #Print the results
    print('Method transmittedPowerCorrection has been called.\nCorrected values of Pt to satisfy the minimal FFM condition:\nPt = ' + 
          str(np.round(Pt, 2)) + 
          '\nValues in text are rounded to 2 decimals, but are stored as np.float64. \nCalculate FFM again with corrected values of Pt.\n\n\n')
    return Pt

def carrierInterferenceRatio(Prx, FB, CI_min):
   
    #Create arrays of Prx and FFM's coming from the left side of the reciever
    #on scheme by adding 0 as the first element.
    FFM_left =  np.concatenate(([0], FFM_corr)) 
    Prx_left = np.concatenate(([0], Prx))
    print(FFM_left)
    print(Prx_left)
    #Create arrays of Prx and FFM's coming from the rightside of the reciever
    #on scheme by adding 0 as the last element.
    Prx_right = np.concatenate((Prx, [0]))
    FFM_right =  np.concatenate((FFM_corr, [0]))
    print(Prx_right)
    print(FFM_right)
    #Create an array for which every station (reciever) will contain two values, 
    #interference from the left side and from right side
    CI = np.zeros((len(d)+1, 2))
    
    #Calculate C/I [dB] for the worst case scenario i.e. maximum FFM on carrier and 
    #minimum (0) FFM on interference
    i = 0
    for val in CI:

        CI[i] = [(Prx_left[i] - FFM_left[i]) - (Prx_right[i] - FB), (Prx_right[i] - FFM_right[i]) - (Prx_left[i] - FB)]
        i = i + 1

    #Leave out first and last element, since they represent end stations and there is
    #no interference from other side
    CI = CI[1:-1]  
    
    #Create an array of booleans for C/I higher than minimum C/I
    CI_cond = CI > CI_min
    
    #Store the results in dict
    result = {'C/I_from_right[dB]':np.round(CI[:, 1],2),
              'C/I_from_left[dB]':np.round(CI[:, 0],2),
              'Satisfies_right':CI_cond[:,1],
              'Satisfies_left':CI_cond[:,1]
              }
    
    #Create a df from dict and return it
    Results = pd.DataFrame(result)
    Results.index = np.arange(2, len(Results)+2)
    Results.index.name = 'Repeater'
    return Results


#Specify noise number [dB]
F_rx = 3

#Specify bandwidth. Units are specified as function arguments.
B = 28

#Specify antennas gains [dBi]. If you use the same antenna for every link, specify only one. 
G = 38.9

#Distance between two links [km]
d = np.array([13.07, 26.52, 14.69, 16.95, 13.92, 18.86, 20.46, 7.24])

#Specify nominal transmitted power [dBm] this script will correct it for each hop
Pt = 20

#Frequency [GHz]
Carrier_freq = 4

Link_names = ["ŽnH-Kunešov", "Kunešov-Tlstá", "Tlstá-Katova_skala", "Katova_skala-Rakytie", "Rakytie-Terchová", "Terchová-Kubínska", "Kubínska-Javorový_vrch", "Javorový_vrch-Trstenná"]

#Set minimum reserve for FFM by calling a method
min = setMinimumFFM(d)

#Calculate the power threshold
Pte6 = powerThreshold(F_rx, B, "128-QAM", "MHz")

#Calculate FFM
FFM = flatFadeMargin(G, d, Pt, Carrier_freq,  Pte6)

#Perform transitted power correction to satisfy the minimum FFM values.
Ptx_corr = transmittedPowerCorrection(Pt, FFM, d, min)

#Calculate FFM for corrected values of Pt
FFM_corr = flatFadeMargin(G, d, Ptx_corr, Carrier_freq,  Pte6)


#Create a dict with all variables you want to print
results = {'Link_names':Link_names,
           'Former_Ptx[dBm]':np.full(len(d), Pt),
           'Distance[km]':d,
           'Former_FFM[dB]':np.round(FFM,2),
           'Corrected_Ptx[dBm]':np.round(Ptx_corr,2),           
           'CorrectedPtx_FFM[dB]':np.round(FFM_corr,2)
    }
#Create a pandas dataframe from specified dict
R = pd.DataFrame(data = results)

#Print the whole dataframe
print(R.to_string())

#Calculate effective isotropic radiated power
P_rx = Ptx_corr- (20*np.log10(d)+20*np.log(Carrier_freq) + 92.4) + 2*G

#Specify front/back ratio
FB_ratio = 69

#Specify minimum C/I ratio
minimum_CI = 15

#Call method to calculate C/I 
CI = carrierInterferenceRatio(P_rx, FB_ratio, minimum_CI)

print(CI.to_string())



