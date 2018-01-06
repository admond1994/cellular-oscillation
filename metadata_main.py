# -*- coding: utf-8 -*-
import numpy as np
from decimal import Decimal
import scipy.stats as stats
import matplotlib.pyplot as plt

#[x] == Calculate mean and std for each txt. file 
#[x] == Separate the periods into one period (Create a period separator function)
#[x] == Calculate the mean period 
#[x] == Obtain the amplitude for each period 
#[x] == Calculate CV for each cell chain 

#The interval between 2 frames is 20mins
#The desired mean period is 5.5 hours(330 minutes)
#The limit cycle is an oscillation with a period of 5.5 h 
#Show agreement with the limit cycle period experimentally
#============================================================================#
def fft(period):
    '''
    This High Frequency Pass Filter passes a frequency higher than a specified
    cutoff frequency and attenuates signals with frequencies lower than the 
    cutoff frequency
    
    Input => period collected in array
    Output => True 
       
    '''
    f = 1/330 #Theoretical frequency
    f_cutoff = f/3 #Cutoff frequency is set to be 3 times lower of the values
                   #Mean slow shift period = 1265 mins
                   #Threshold to filter ~ 1000 mins
    if len(period) != 0: #Check the period is not empty array
        f_test = 1/period[0] #Input frequency
        if f_test < f_cutoff: #If the frequency is smaller than cutoff frequency
            return True
        else:
            return False

#============================================================================#
def period_separator(time,fluo_intensity):
    '''
    This function separates periods into one period
    To select a period --> fulfil the specified boundary conditions
    Output --> Collected periods
    
    Boundary Conditions::
    
    1. Cross threshold_up from below
    2. Cross threshold_down from above

    Features of a period:: 
    
    1. Shape - Up Down Up
    2. A time before and after are only captured after the line hits data_mean
    '''       
    c = 0.16
    data_std = np.std(fluo_intensity)
    data_mean = np.mean(fluo_intensity)
#    print('mean:{}'.format(data_mean))
    threshold_up = data_mean + (c*data_std)
    threshold_down = data_mean - (c*data_std)
#    print('threshold_up:{}'.format(threshold_up))
#    print('threshold_down:{}'.format(threshold_down))

    time_frame = np.array([])
    time_frame_update = np.array([])
    final_period = np.array([])
    spike_collect_final = []
    
    #Conditions are fulfilled when the total count is 2
    count_1 = 0
    count_2 = 0
    
    for i in range(1,len(fluo_intensity)):
#        print('Now is:{}'.format(i))
            #1st condition     
            #Consider the region above upper boundary
        if (fluo_intensity[i-1] < threshold_up < fluo_intensity[i]): #if positive gradient 
            if count_1 == 1: #A timeframe has been captured, this is to avoid recapture of other time frame
#                print('                    continue')
                continue
            test_count_1 = 0
            test_1 = fluo_intensity[i]
            while test_1 > data_mean:
                test_count_1 = test_count_1 + 1
                test_1 = fluo_intensity[i - test_count_1]
                if time[i-test_count_1] == 0: #If the test_1 has reached the first index, break the loop
                    break
               
                if test_1 < data_mean:   
                    count_1 = 1 #If a previous point intersects with mean boundary,1st condition has been fulfilled
                    #Update timeframe      
                    start_point = i - test_count_1
                    time_frame = np.concatenate((time_frame,np.array([time[start_point]])))
#                    print('time_frame_update:{}'.format(time_frame))
                    break                    
        #2nd condition
        #Consider the region below lower boundary
        elif fluo_intensity[i] < threshold_down < fluo_intensity[i-1]: #If ther gradient is negative and intersects with lower boundary
            if count_1 != 1:
                count_2 = 0 #Got to fulfil 1st condition before 2nd condition
                continue
            else:
                test_count_2 = 0
                test_2 = fluo_intensity[i]
                
                while test_2 < data_mean:
                    test_count_2 = test_count_2 + 1
                    if i+test_count_2 >= len(time)-1:
                        break
                    else:
                        test_2 = fluo_intensity[i + test_count_2]
                    if test_2 > data_mean:
                        count_2 = 1
                        end_point = i + test_count_2
                        time_frame = np.concatenate((time_frame,np.array([time[end_point]])))
                        time_frame_update = np.concatenate((time_frame_update,time_frame))
#                        print('time_frame_update_FINAL:{}'.format(time_frame_update))
                        break
        else: 
            pass
        #If count_total = 2,two conditions are fulfilled
        #Separate the period timeframe
        #final_period is used to collect all the separated periods in an array
        count_total = count_1 + count_2                            
        if count_total == 2:
            time_collected = time[np.arange(start_point,end_point+1)] 
#            print(time_collected)
            fluo_intensity_collected = fluo_intensity[np.arange(start_point,end_point+1)]
#            print(fluo_intensity_collected)
            
            period = np.diff(time_frame_update)
#            print('check')
            if fft(period) == True: #frequency is smaller than cutoff frequency
                period = np.array([])
            else:
                spike_collect = spikeCollect(time_collected,fluo_intensity_collected)
                spike_collect_final.append(spike_collect)
            final_period = np.concatenate((final_period,period))
            time_frame = np.array([])
            time_frame_update = np.array([])
            count_1 = 0
            count_2 = 0

    if len(final_period) == 0: #If the final_period is an empty array s
        final_period = np.array([0])
    return final_period,spike_collect_final
    

#============================================================================#
def cell_chain_mean_period(period_appended):
    '''
    Input => Appended period for all the cell chains
    Output => Final mean value of all the cell chains
    
    '''
    
    mean_period_chain = Decimal(np.mean(period_appended))
    mean_period_chain_FINAL = round(mean_period_chain,2)
    
    return mean_period_chain_FINAL 
#============================================================================#
def spikeCollect(targetTime,target):
    '''
    Input -> (time_array, target_array), both are arrays
    Output -> [[spike_amp,spike_pos]]
    '''
    spikeStore = []
    mean = np.mean(target) #l_sensor = mean
    stdv = np.std(target)
    h_sensor = mean + 0.16*stdv
    local_max = np.max(target)
#    spike_pos = np.argmax(target)
    spike_amp = local_max - mean
    
    if local_max > h_sensor: #local_max must be bigger than upper threshold
        spikeStore.append(spike_amp)
    return spikeStore
#============================================================================#
def collect_cv(spike_data):
    '''
    Input => Spike amplitude data for each cell
            [Spike_amp]
    Output => Appended list of CV for all the cells    
    '''
    a = spike_data
    collect_cv = []
    for i in range(len(a)):
        if a[i] == []:
            collect_cv.append(0)
        else:
            cell_cv_array = stats.variation(a[i])
            cell_cv = cell_cv_array[0]
            collect_cv.append(cell_cv)
    
    return collect_cv
    

#============================================================================#
def spikeCollect_Alternative(targetTime,target):
    """
     extract all spikes from time trajectory for each period
     Input -> (time_array, target_array), both are arrays:
     Output-> [count, [[spike_amp, spike_peak, spike_right_edge]...]]
  
      spike_amp = localMax-mean
      spike_peak = time where the spike occurs 
      spike_right_edge = end of spike
    """
    marker=[False,True]
    localMax=0.
    maxInstant=0.
    count=0
    spikeStore=[]
    mean= np.mean(target)
    stdv=np.std(target)
    l_sensor=mean
    h_sensor=mean+0.16*stdv
    i_object=(np.array([targetTime[:-1],target[:-1],target[1:]])).transpose()
    for t,sl,sr in i_object:
        if marker[1]==True :
            localMax=max(localMax,sl)
            maxInstant=t if localMax==sl else maxInstant
            if sl<l_sensor and sr>l_sensor:
                if marker[0]==True :
                    count+=1
                    spikeStore.append([localMax-mean,maxInstant,t])
                marker[1]=False
                localMax=0
                marker[0]=True
        elif sl<h_sensor and sr>h_sensor and marker[1]==False :
            marker[1]=True
    return spikeStore
#============================================================================#
def collect_cv_Alternative(spike_data):
    '''
    Input => Spike data for each cell
            [[spike_amp, spike_peak, spike_right_edge]...]]
            
    Output => Appended list of CV for all the cells
    '''
    a = spike_data    
    amp_collect = [] 
    cv_collect = []

    for i in range(len(a)):
        if i > 0: #Collect the previous list of amplitudes and calculate CV 
            if amp_collect == []: #the cell has empty array, no amplitudes are collected
                cv_collect.append(0)
            else:
                cv_collect.append(stats.variation(amp_collect)) #Calculate CV
                amp_collect = [] #Collect the next cell amplitudes with empty list
        if len(a[i]) != 0:
            for j in range(len(a[i])): #Specified cell
                if a[i][j] != []:
                    for k in range(len(a[i][j])):
                        if a[i][j][k] != []:
                            for l in range(len(a[i][j][k])):
                                if l == 0:
                                    amp_collect.append(a[i][j][k][l]) #Append the list with amplitudes
                                    if i == 24 and j == 4:
                                        cv_collect.append(stats.variation(amp_collect))
                    
    return cv_collect
            

#============================================================================#
def evalStat(data):
    '''
    Input => spike amplitudes
    Output => mean,std, and CV of spike amplitudes
    '''
    mean=np.mean(data)
    std=np.std(data)
    CV=stats.variation(data)
    return [mean, std, CV]
#============================================================================#
#MAIN PROGRAM

collect_mean_period_cell_chain = []
spike_data_collect = []


time_frame = np.array([])
#Get the array of time
for i in range(0,4340,20):
    time_frame = np.concatenate((time_frame,np.array([i])))
    
data = np.loadtxt('p53 for position1.txt', delimiter='\t', unpack=True)
data_transpose = data.T

for i in range(len(data_transpose)):
    cell_chain = data_transpose[i]
    final_period,spike_data = period_separator(time_frame,cell_chain)
    spike_data_collect.append(spike_data) #Append spike data for every cell
                                          #Total length should be 25 with 3 elements in each index
    
    print('Cell_{}\nCollected periods:{}\n'.format(i+1,final_period))
    if (len(final_period) == 1 and final_period[0] == 0):
        continue

    collect_mean_period_cell_chain.append(np.mean(final_period))
#    print('Cell:{}\nPeriod collected:{}\n'.format(i,final_period))

mean_period_all_chains = cell_chain_mean_period(collect_mean_period_cell_chain)
#count, spikeStore = spikeCollect_Alternative(time_frame, cell_chain)
#print('count:{}\nspikeStore:{}\n'.format(count,spikeStore))

print('Collected mean period cell chains:\n{}\n\nOverall mean period:{}\n'\
.format(collect_mean_period_cell_chain,mean_period_all_chains))

cv_data = collect_cv(spike_data_collect)

cell_size = [508.600,543.182,570.571,583.636,614.364,628.571,638.286,688.875,730.000,839.455,865.625,870.600,895.000,929.889,973.000,1096.000,1276.500]
cv = [0.638,0.338,0,0.375,0.712,0.361,0.586,0.730,0.162,0,0.145,0.424,0.589,0.070,0.527,0,0.189]

plt.plot(cell_size,cv,'ro')
plt.xlabel('Cell Size')
plt.ylabel('CV')
plt.title('CV vs Cell Size')
plt.show()




