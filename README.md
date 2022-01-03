# Microwave-P2P-link
Script to calculate Pthe-6, minimum FFM and minimum Ptx for each link. Script utilizes methods and is designed according to this diagram:

![ezcv logo](/Diagram.png)


# Functions and order in which to use them

Use

```
setMinimumFFM(d)
```

to set minimum FFM for each link based on distance. 

```
powerThreshold(F_rx, B, "M-QAM", "xHz")
```

to calculate Pte for link. Arguments are explained inside of the function. Use calculated Pte inside next function.

```
flatFadeMargin(G, d, Pt, Carrier_freq, Pte)
```

to calculate FFM for BER-3, -6. Arguments are specified inside of the function. Use

```
transmittedPowerCorrection(Pt, FFM, d) 
```

to correct transmitted power in order to satisfy minimum FFM for BER-6. Use **flatFadeMargin** again to calculate FFM. The corrected values should be equal to minimal values of FFM. 

Use 
```
carrierInterferenceRatio(Pt, FB, 15, FFM_corr, G, Carrier_freq)
```

to calculate C/I and check if the values satisfy minimal condition. Link draw and quality are calculated by following functions:
```
Quality = linkQuality(B, FFM, Carrier_freq, d)
Draw = linkDraw(Quality, d)
```

Afterwards, following power correction is neccesary using **drawPowerCorrectoin**:
```
condition = Draw['is_>100%'].sum()

#Do the power correction
Corrected_power = drawPowerCorrection(Draw, Pt)

#Correct the power using Corrected_power as argument passed in methods inside while loop until all values in Draw DF are >100%
while condition != 0:
    
    #Calculate FFM
    FFM = flatFadeMargin(G, d, Corrected_power, Carrier_freq,  Pte)

    #Calculate quality and draw
    Quality = linkQuality(B, FFM, Carrier_freq, d)
    Draw = linkDraw(Quality, d)

    #Recurrently update value of condition
    condition = Draw['is_>100%'].sum()

    #Check for link power levels that still need correction and raise them by 0.5
    Corrected_power = drawPowerCorrection(Draw, Corrected_power)
    
    #Print message
    print(str(condition) + ' values still left to correct to satisfy draw condition.')
    
#Print the once condition is satisfied
else: print('Power is corrected')
```

Each iteration, conditions are checked and OR operation is perfrormed multiple times until one Bool is at the output. If the bool is **True (1)**, it means that there is at least one hop which doesnt satisfy given condition. If the bool output is **False (0)** it means that all conditions are satisfied. Once power is corrected, use **flatFadeMargin** and **carrierInterferenceRatio** again. 

This script was designed to start off with low Pt, code will then correct the values twice. Code inside of the repository starts with Pt = 10 dBm. 
