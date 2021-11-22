import numpy as np
import math
import pandas as pd
from scipy.special import erfc

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
    SNR_e6 = {'4-QAM':13.2,
              '8-QAM':16.90,
              '16-QAM':20.5,
              '32-QAM':23.4,
              '64-QAM':26.6,
              '128-QAM':29.5,
              '256-QAM':32.6}

    SNR_e3 = {'8-QAM':12.08,
              '16-QAM':15.39,
              '32-QAM':18.55,
              '64-QAM':21.63,
              '128-QAM':24.67,
              '256-QAM':27.70}

    #Power threshold for BER-6
    Power_threshold = {'BER_e6':[Pn + F_rx + SNR_e6[M_QAM]],
                       'BER_e3':[Pn + F_rx + SNR_e3[M_QAM]]}
    
    P_th = pd.DataFrame(Power_threshold)

    #Print the dataframe
    print("Power threshold is:\n" + str(P_th) + "\n")
    return P_th
    



def flatFadeMargin(Ga, d, Ptx, fc, Pthe):
    '''
    Calculates FFM for each link using given arguments. Function assumes using same antenna on every link.

    \nGa:antenna gain. Probably of type double.
    \nd: array of distances for each link.
    \nPtx: Transmitted power. Single value. Could be int, double or float.
    \nfc: Noise number. Assuming ideal transmitter it is 0. Usually int but can be double or float.
    \nPthe6: Power threshold for BER-6. Calculate using powerThreshold.

    '''
    Pth_e6 = Pthe.loc[:, 'BER_e6']
    Pth_e3 = Pthe.loc[:, 'BER_e3']

    Pe6 = Pth_e6.to_numpy()
    Pe3 = Pth_e3.to_numpy()

    
    #Create an array of length d full of Ptx and float 64 type
    Pt = np.full(len(d), Ptx, dtype = np.float64)
    
    #Calculation of effective isotropic radiated power
    P_eirp = Pt + Ga
    
    #Calculate link fade
    A_los = 20*np.log10(d)+20*np.log(fc) + 92.4
    
    #Calculate power on the reciever antenna
    P_rx = P_eirp - A_los + Ga

    
    flatFade = {'BER_e6':P_rx - Pth_e6[0],
                'BER_e3':P_rx - Pth_e3[0]}

    FFM = pd.DataFrame(flatFade)

    print("FFM is:\n" + str(FFM) + "\n")
    return FFM
        

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
    FFM_e6 = FFM.loc[:, 'BER_e6']
    #Create an np array of length d and values of Pt at each places
    Pt = np.full(len(d), Ptx, dtype = np.float64)

    #Create an array of booleans for given condition
    Ptx_bad = FFM_e6 < min

    #Find indices of values which satisfy the given condition
    indices = [i for i, x in enumerate(Ptx_bad) if x]

    #For each index which satisfies given condition, update the value by given formula
    for i in indices:
        Pt[i] = Pt[i] + (min[i] - FFM_e6[i])
    
    #Print the results
    print('Method transmittedPowerCorrection has been called.\nCorrected values of Pt to satisfy the minimal FFM condition:\nPt = ' + 
          str(np.round(Pt, 2)) + 
          '\nValues in text are rounded to 2 decimals, but are stored as np.float64. \nCalculate FFM again with corrected values of Pt.\n\n\n')
    return Pt
 

def linkQuality(B, FFM, f, d):
    #Calculate fmin and fmax
    f1 = -B/2
    f2 = B/2

    #Calculate signature
    S = np.exp(-(f1 + f2)/7.6 * (6))

    c = 0.065
    S_e3 = S
    S_e6 = S * 1.7
    tau = 6.3
    q1 = (-1.5 * d / 100) + 1.1
    q2 = 0.5

    #Probability of interferential fading occurence
    PR = (c * Carrier_freq * 10**-3 * d**3 * 6 * 10**(-8))

    #Probability of selective fading occurence
    Ps = (1 - np.exp(-6.3 * 10**(-3) * PR**0.75))

    #Mean value of delay
    tau_er = 0.7 * (d/50)**1.3

    #BER-3 error rate 
    t3sM = 9.25 * 10**(-3) * Ps * S_e3 * (tau_er**2 / tau) * q1
    t3sN = 9.25 * 10**(-3) * Ps * S_e3 * (tau_er**2 / tau) * (1 - q1)
    t3s = t3sM + t3sN
    
    #BER-6 error rate
    t6sM = 9.25 * 10**(-3) * Ps * S_e6 * (tau**2 / tau) * (q1)
    t6sN = 9.25 * 10**(-3) * Ps * S_e6 * (tau**2 / tau) * (1 - q1)
    t6s = t6sM + t6sN
    
    FFM_3 = FFM['BER_e3'].to_numpy()
    FFM_6 = FFM['BER_e6'].to_numpy()

    t3f = 10**(-0.1*FFM_3)
    t6f = 10**(-0.1*FFM_6) 


    #Calculation of probability that t >= 10
    numerator = (10 - 10 * np.log10((56.6 * np.sqrt(d))) /  (Carrier_freq) * 10**(-(FFM_3 / 20)))
    denominator = (7.55 - c * FFM_3) * np.sqrt(2)

    Prob_t10 = (1/2 * erfc(numerator / denominator + 0.375))

    #Calculation of idle time by flat, interferential fadings
    t_Kf = t3f/4 * Prob_t10

    #Calculation of idle time by selective fadings
    t_Ks = t3s/4 * Prob_t10
    t_K = t_Kf + t_Ks

    #SES calculation by interference fadings
    t_SESf = t3f * (1 - Prob_t10)

    #SES calculation by selective fadings
    t_SESs = t3s * (1 - Prob_t10)

    t_SES = t_SESf + t_SESs

    #DM calculation by flat, interferential fadings
    t_DMf = 5.5 * (t6f - t3f)

    #DM calculation by selective fadings
    t_DMs = 5.5 * (t6s - t3s)

    #DM calculation
    t_DM = t_DMf + t_DMs
    
    #Store results as a dict
    Q = {'Idle_time':t_K,
         'SES':t_SES,
         'DM':t_DM}

    #Create pandas dataframe from Q and return it
    Quality = pd.DataFrame(Q)
    print("Link quality is:\n" + str(Quality) + "\n")
    return(Quality)

def linkDraw(Q, d):
    
    idleTime = Q['Idle_time'].to_numpy()
    SES = Q['SES'].to_numpy()
    DM = Q['DM'].to_numpy()


    idleTime_perc = (0.0112/280)*d
    SES_perc = (0.006/280)*d
    DM_perc = (0.0448/280)*d
        
    d = {'Idle_time': idleTime * 100/idleTime_perc,
         'SES':SES * 100/SES_perc,
         'DM':DM * 100/DM_perc }

    Draw = pd.DataFrame(d)
    print("Link draw is:\n" + str(Draw) + "\n")
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

#Specify link names
Link_names = ["ŽnH-Kunešov", "Kunešov-Tlstá", "Tlstá-Katova skala", "Katova skala-Rakytie", "Rakytie-Terchová", "Terchová-Kubínska", "Kubínska-Javorový vrch", "Javorový vrch-Trstenná"]

#Set minimum reserve for FFM
min = setMinimumFFM(d)

#Calculate the power threshold
Pte = powerThreshold(F_rx, B, "128-QAM", "MHz")

#Calculate FFM
FFM = flatFadeMargin(G, d, Pt, Carrier_freq, Pte)

#Perform transitted power correction to satisfy the minimum FFM values.
Pt = transmittedPowerCorrection(Pt, FFM, d)

#Calculate FFM for corrected values of Pt
FFM = flatFadeMargin(G, d, Pt, Carrier_freq,  Pte)

#Calculate idle time, SES and DM
Quality = linkQuality(B, FFM, Carrier_freq, d)

Draw = linkDraw(Quality, d)
