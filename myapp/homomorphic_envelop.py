import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import  butter,filtfilt
from scipy.signal import hilbert
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from scipy.signal import resample

def butter_bandpass_filter(data, low, high, sampling_rate, order=2):
    nyquist = 0.5 * sampling_rate
    low = low / nyquist
    high = high/nyquist
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y

def Homo_Env(sig, sr):
    Env = abs(hilbert(sig))
    # return Env
    Homo_env =  np.exp(butter_bandpass_filter(np.log(Env), 0.5, 8,sampling_rate= sr,order = 2))
    # Homo_env =  butter_bandpass_filter(np.log(Env), 0.5, 8,sampling_rate= sr,order = 2)
    return Homo_env

def get_systolic_duration(correlation_signal, heart_cycle_duration):
    minimum_location = 200
    maximum_location = heart_cycle_duration // 2
    locs, _ = find_peaks(correlation_signal, height= 0)
    peaks = correlation_signal[locs]
    new_location = locs[(locs >= minimum_location) & (locs <= maximum_location)]
    new_peaks = peaks[(locs >= minimum_location) & (locs <= maximum_location)]
    if len(new_location) == 0:
        return -1
    peak_indices = np.argsort(new_peaks)[::-1]
    first_peak_location = new_location[peak_indices[0]]
    # print(new_location)
    # print(new_peaks)
    return first_peak_location

def Heart_Rate(Homo_env, sampling_rate):
    minimum_location = int(0.5 * sampling_rate)
    maximum_location = int(1.8 * sampling_rate)
    Heart_rate = []
    systolic_duration = []
    if len(Homo_env) >= 3*sampling_rate:  # if i have at least buffer of 3 second
        N = (len(Homo_env) - 3*sampling_rate) // sampling_rate
        for i in range(N-1):
            segment = Homo_env[i*sampling_rate : (i+3)*sampling_rate] # Take 3 second duration segment
            correlation = np.correlate(segment, segment, mode= 'full')
            corr_corrected = correlation[len(segment):len(correlation)]
            locs, _ = find_peaks( corr_corrected,  height=0)
            peaks = corr_corrected[locs]
            new_location = locs[(locs >= minimum_location) & (locs <= maximum_location)]
            new_peaks = peaks[(locs >= minimum_location) & (locs <= maximum_location)]
            # first_peak_location = new_location[np.argmax(new_peaks)]
            # new_location[np.argmax(new_peaks)] = False
            # new_peaks[np.argmax(new_peaks)] = False
            # second_peak_location = new_location[np.argmax(new_peaks)]
            
            peak_indices = np.argsort(new_peaks)[::-1]
            
            if len(peak_indices) == 0:
                continue
            first_peak_location = new_location[peak_indices[0]]
            
            if len(peak_indices) == 1:
                second_peak_location = first_peak_location
            else:
                second_peak_location = new_location[peak_indices[1]]
            
            if first_peak_location / second_peak_location >= 1.85 and first_peak_location / second_peak_location <= 2.2:
                peak_location = second_peak_location
            else:
                peak_location = first_peak_location
            Heart_rate.append(peak_location) 
            
            systolic_location = get_systolic_duration(correlation_signal= corr_corrected, heart_cycle_duration= peak_location)
            if(systolic_location != -1) :
                systolic_duration.append(systolic_location)
                
                # print(systolic_duration)
                # print(np.min(systolic_location))
            # plt.plot(corr_corrected)
            # plt.text(0.9, 0.6, f'HCD  {peak_location}', color='red', transform=plt.gca().transAxes, fontsize=10)
            # plt.text(0.9, 0.5, f'Silent Systole  {systolic_location}', color='blue', transform=plt.gca().transAxes, fontsize=10)
            # plt.text(0.9, 0.4, f'Systole  {systolic_location - 122}', color='blue', transform=plt.gca().transAxes, fontsize=10)
            # plt.text(0.9, 0.3, f'Diastole  {peak_location - 92 - systolic_location}', color='blue', transform=plt.gca().transAxes, fontsize=10)
            
            # plt.show()
            # plt.savefig(f'{i}_corr.png')
            # plt.clf()
            
            # plt.plot(segment)
            # plt.show()
            # plt.savefig(f'{i}_seg.png')
            # plt.clf()
    return Heart_rate, systolic_duration

def getHeartRate(original_signal):
    sampling_rate = 1000
    resampling_factor = sampling_rate / 50
    signal = resample(original_signal, int(len(original_signal) * resampling_factor))
    # print(len(signal))
    filtered_signal = butter_bandpass_filter(signal, 20,180, sampling_rate, order=4)
    filtered_signal = filtered_signal - filtered_signal.mean()
    filtered_signal = filtered_signal / np.max(abs(filtered_signal))
    homomorphic_envelope = Homo_Env(sig= filtered_signal, sr= sampling_rate)
    homomorphic_envelope_normal = minmax_scale(homomorphic_envelope)
    HR = Heart_Rate(Homo_env= homomorphic_envelope_normal, sampling_rate= sampling_rate)
    return HR
    
if __name__ == '__main__':
    audio_file = "//home3/luharj/AiSteth/Training_Data/Circor/Normal Signal/43852_AV.wav"
    print(getHeartRate(audio_file= audio_file))
    