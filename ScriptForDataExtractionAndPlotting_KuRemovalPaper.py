# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:52:03 2025

@author: Pradeep Sathyanarayana, Loparo Lab, Dept of BCMP, Harvard Medical School
pradeep_sath@hms.harvard.edu

Script for extracting raw traces from 'pickle' databases for single-molecule data.
Intensity traces from each condition described in the manuscript are deposited as individual
zipped files. Extract all the .pkl files corresponding to a condition in to a folder.

Copy folder path and initialize a variable named 'path' as follows  :
    path = r'<copied path of the folder containing pickle files (for one experimental condition)>'

I have broken the code into Blocks. Each block is lines of code between two "#%%" symbols. If you use Spyder,
these automatically show up as blocks. 

Some remarks about the variables in the code:

1. greenChTracks - refers to single-molecule trajectories of molecules from the greenChannel. In all experiments
described in the manuscript, they represent Ku trajectories.

2. redChTracks - refers to single-molecle trajectories of molecules from the red channel. In all experiments
described in the manuscript, they represent fluorescently labeled DNA (either 5' or 3')   

3. stepFitGreen/stepFitRed - refers to fits to raw trajectories to identify the step-loss of intenisty. 
This was used to compute liftime of Ku or 5' DNA and the survival plot. These can be plotted to 
assess how robustly the point of loss was determined.
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
%matplotlib auto


#%% Run this block of code first to load all the file names in to python
# Block 2
os.chdir(path)
fileList = os.listdir(path)
# Getting list of pkl files from folder
tifFiles = []
for i in fileList:
    if i.endswith('tifgreenChTraces_Only.pkl'):
        tifFiles = np.append(tifFiles,i)

movieList =[i[:-26] for i in tifFiles]  # Getting fileNames from which data was acquired

#%% Here all the green and red channel tracks are collated from all the movies from which single-molecule trajectories were recorded.
# Block 3
dataframesGreen = [pd.read_pickle(str(f)+".tifgreenChTraces_Only.pkl") for f in movieList]
#  arranging the data in accessible manner so that it can be plotted.

for n in range(len(dataframesGreen)):
    temp = dataframesGreen[n]['greenTime']
    temp2 = [temp.iloc[0]] * len(temp)  # Create a list with the first value repeated
    dataframesGreen[n]['greenTime'] = temp2
    
combined_Green_df2 = pd.concat(dataframesGreen, ignore_index=True)


dataframesRed = [pd.read_pickle(str(f)+".tifredChTraces_Only.pkl") for f in movieList]
#  arranging the data in accessible manner so that it can be plotted.

for n in range(len(dataframesRed)):
    tempvar = dataframesRed[n]['redTime']
    tempvar2 = [tempvar.iloc[0]] * len(tempvar)  # Create a list with the first value repeated
    dataframesRed[n]['redTime'] = tempvar2

combined_Red_df2 = pd.concat(dataframesRed, ignore_index=True)


# collating all the greenChTracks and redChTracks into single array.
greenChTracks_all = [combined_Green_df2.greenChTracks[i] for i in range(len(combined_Green_df2.greenChTracks))]
stepFitGreen_all = [combined_Green_df2.stepFitGreen[i] for i in range(len(combined_Green_df2.stepFitGreen))]

redChTracks_all = [combined_Red_df2.redChTracks[i] for i in range(len(combined_Red_df2.redChTracks))]
stepFitRed_all = [combined_Red_df2.stepFitRed[i] for i in range(len(combined_Red_df2.stepFitRed))]



redTime_all = [combined_Red_df2.redTime[i] for i in range(len(combined_Red_df2.redTime))]
# redTime_all = [redTime_all[0] for i in redTime_all]

greenTime_all = [combined_Green_df2.greenTime[i] for i in range(len(combined_Green_df2.greenTime))]
# greenTime_all = [greenTime_all[0] for i in greenTime_all]


#%% The code in the following block plots the greenChTracks and redChTracks. 16 molecules are plotted
#in each subplot. 
# Block 4    
ticker = 0 
handles = []  # List to store the plot handles for the global legend
labels = ['Ku', 'DNA']   # List to store the corresponding labels for the global legend

while ticker < len(greenChTracks_all):
    
    fig, axs = plt.subplots(4, 4)
    
    for i in np.arange(0, 4):
        for j in np.arange(0, 4):
            
            # Plotting data and creating individual legend entries
            line1, = axs[i, j].plot(greenTime_all[ticker], greenChTracks_all[ticker], '-C2', label='Ku')                            
            line2, = axs[i, j].plot(greenTime_all[ticker], stepFitGreen_all[ticker], '-C2') 
            line3, = axs[i, j].plot(redTime_all[ticker], stepFitRed_all[ticker], '-C3') 
            line4, = axs[i, j].plot(redTime_all[ticker], redChTracks_all[ticker], 'C3-', label='Red Channel')
            
            # Adding plot handles for the global legend
            handles.append(line1)
            handles.append(line4)
            
            axs[i, j].set_title('Track No.' + str(ticker))
            ticker += 1
    
    # Set labels for the whole figure
    fig.supxlabel('Time (s)', fontsize=14)
    fig.supylabel('Intensity (arb. units)', fontsize=14) 
    plt.subplots_adjust(top=0.85)  # Adjust the top space to allow for the global legend

    # Add a global legend at the top of the 4x4 grid (above the subplots)
    fig.legend(handles=[handles[0], handles[1]], labels=labels, loc='upper center', ncol=2, fontsize=16)

    # Set figure size
    fig.set_figheight(15)
    fig.set_figwidth(25)

    plt.show() 
    
    
#%% The code in the following block is to plot the survival probability curves for the Ku (greenCh) and DNA (redChannel)
# # Block 5  
kuOnTime = []
dnaOnTime = []
dnaCensorship = []
kuCensorship=[]

for kk in movieList:
    dfTempGreen = pd.read_pickle(str(kk)+".tifgreenChTraces_Only.pkl")
    kuOnTimeTemp = dfTempGreen['kuOnTime']
    kuOnTime = np.append(kuOnTime,kuOnTimeTemp)
    dfTempRed = pd.read_pickle(str(kk)+".tifredChTraces_Only.pkl")
    dnaOnTimeTemp = dfTempRed['dnaOnTime']
    dnaOnTime = np.append(dnaOnTime,dnaOnTimeTemp)
    
    dnasensorshipTemp = np.empty([len(dnaOnTimeTemp),1])
    for n,i in enumerate(dnaOnTimeTemp):
        if i == np.max(dnaOnTimeTemp):
            dnasensorshipTemp[n] = 0
        else:
            dnasensorshipTemp[n]=1
    dnaCensorship = np.append(dnaCensorship,dnasensorshipTemp)
       
    kusensorshipTemp = np.empty([len(kuOnTimeTemp),1])
    
    for n,i in enumerate(kuOnTimeTemp):
        if i == np.max(kuOnTimeTemp):
            kusensorshipTemp[n] = 0
        else:
            kusensorshipTemp[n]=1
    kuCensorship = np.append(kuCensorship,kusensorshipTemp) 
    
# Block 5 
plt.figure()
fig, ax = plt.subplots(figsize= (7, 5.5))
kmf = KaplanMeierFitter()
kmf.fit(dnaOnTime, dnaCensorship,label='DNA_Lifetime  [N ='+ np.str(len(dnaOnTime))+']')
kmf.plot(ci_show=True)
kmfku = KaplanMeierFitter()
kmfku.fit(kuOnTime, kuCensorship,label='Ku_Lifetime  [N ='+ np.str(len(kuOnTime))+']')
kmfku.plot(ci_show=True)
ax.set_ylabel("Survival probability", size=18)
ax.set_xlabel("Time (s)", size=18)
 ## ci_show is meant for Confidence interval, since our data set is too tiny, thus i am not showing it.
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.spines["left"].set_linewidth(1)

ax.spines["bottom"].set_linewidth(1)

ax.spines["top"].set_linewidth(1)

ax.spines["right"].set_linewidth(1)

DNAmedian_ci = median_survival_times(kmf.confidence_interval_)
Kumedian_ci = median_survival_times(kmfku.confidence_interval_)
DNAmedianAr = DNAmedian_ci.to_numpy()
KumedianAr = Kumedian_ci.to_numpy()

print('Median DNA survival time:', kmf.median_survival_time_, 'CI [',DNAmedianAr[0,0],',', DNAmedianAr[0,1], ']' )
print('Median Ku survival time:', kmfku.median_survival_time_,  'CI [',KumedianAr[0,0],',', KumedianAr[0,1], ']' )
  
