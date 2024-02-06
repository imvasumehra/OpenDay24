from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import spectrogram, windows
import pywt
import librosa
from scipy.signal import resample
from model import getModel
import matplotlib.pyplot as plt


def set_default_parameter():
    slice_length = 3
    downsample_rate = 1000
    label_sampling_rate = 50
    noverlap = 0.5
    padding_value = 0
    append_envelopes = True
    return slice_length,downsample_rate,label_sampling_rate,noverlap,padding_value,append_envelopes

def butter_bandpass(lowcut, highcut, fs, order = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(order, [low, high], btype = 'band')
    return b,a

def butter_lowpass(data, lowcut, fs, order = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, 'low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    b,a = butter_bandpass(lowcut= lowcut, highcut= highcut, fs= fs, order= order)
    y = filtfilt(b, a, data)
    return y

def normalize(x, axis = 1):
    norm_data = np.array(x, dtype= np.float32)
    norm_data = norm_data - norm_data.mean()
    try:
        norm_data = norm_data / norm_data.std()
    except:
        norm_data = norm_data + 0.00001
    return norm_data

def hilbert_env(audio):
    return abs(hilbert(audio))


def homorphic_env(audio, sr):
    audio[audio == 0] = 0.000000001
    log_audio = np.log(abs(audio))
    filtered = butter_lowpass(data= log_audio, lowcut= 10, fs= sr, order= 4)
    exponential_signal = np.exp(filtered)
    return exponential_signal

def PSD_env(audio, sr):
    window_length = int(sr * 0.1)
    hann_window = windows.hann(window_length)
    hop_length = int(sr * 0.02)
    number_of_overlap_samples = int(round(window_length - hop_length)) 
    # print(window_length, hop_length, number_of_overlap_samples)
    audio = np.append(np.zeros(int(number_of_overlap_samples // 2)), audio)
    # print(len(audio))
    f, t, S = spectrogram(audio, sr, window= hann_window, nperseg= window_length, noverlap= number_of_overlap_samples, nfft= 1000)
    PSD = np.zeros(S.shape[1])
    for i in range(S.shape[1]):
        loc = np.argmax(S[20:150, i])
        if S[loc, i] > 0:
            PSD[i] = np.mean(S[loc+15: loc+25, i])
        else:
            PSD[i] = 0
    return PSD


def stationary_wavelets_decomposition(signal_in, wavelet='db4', levels=[2,3,4],
                                      start_level=1, end_level=6, erase_pad=True):
    N = signal_in.shape[0]
    points_desired = 2 ** int(np.ceil(np.log2(N)))
    pad_points = (points_desired-N) // 2
    audio_pad = np.pad(signal_in, pad_width=pad_points, constant_values=0)
    coeffs = pywt.swt(audio_pad, wavelet=wavelet, level=end_level, start_level=start_level)
    wav_coeffs = np.zeros((len(coeffs[0][1]), 0))
    
    for level in levels:
        coef_i =  np.expand_dims(coeffs[-level + start_level][1], -1)
        wav_coeffs = np.concatenate((wav_coeffs, coef_i), axis=1)
    
    if erase_pad:
        wav_coeffs_out = wav_coeffs[pad_points:-pad_points]
    else:
        wav_coeffs_out = wav_coeffs
    return wav_coeffs_out

def get_mfcc_features(audio, sampling_rate, new_length):
    window_length = int(0.1 * sampling_rate)
    hop_length = int(0.02 * sampling_rate)
    window = 'hann'
    power = 2
    n_fft = sampling_rate // 2
    f_min = 20
    f_max = 450
    n_mels = 11
    n_mfcc = 11
    center = True
    mfcc_coeff = librosa.feature.mfcc(y=audio, 
                                      sr=sampling_rate, 
                                      n_mels = n_mels, 
                                      n_mfcc=n_mfcc, 
                                      win_length=window_length, 
                                      hop_length=hop_length, 
                                      window=window, 
                                      n_fft=n_fft, 
                                      fmin=f_min, 
                                      fmax=f_max, 
                                      center=center, 
                                      power=power, 
                                      dct_type=2, 
                                      norm='ortho', 
                                      lifter=0)
    mfcc_coeff_T = np.transpose(mfcc_coeff)
    # print('MFCC shape', mfcc_coeff_T.shape, new_length)
    return np.array(mfcc_coeff_T[0:new_length, :])

def get_delta_mfcc(mfcc_coeff, new_length):
    mfcc_delta = librosa.feature.delta(mfcc_coeff)
    return mfcc_delta[0:new_length,:]


def get_audio_features(audio, sampling_rate= 1000):
    audio[audio == 0] = 1e-10
    audio = normalize(audio)
    new_length = int((len(audio) / sampling_rate) * 50)
    audio_info = np.zeros((new_length , 0))
     
    denoise_audio = butter_bandpass_filter(audio, 20, 150, sampling_rate, order=4)
    
    # ***************** Homomorphic Envelope *********************
    homomorphic_envelop = homorphic_env(audio= denoise_audio, sr= sampling_rate)
    # plot_signal(homomorphic_envelop, 'a')
    normalized_homomorphic_envelop = normalize(x= homomorphic_envelop)
    # plot_signal(normalized_homomorphic_envelop, 'b')
    resampled_homomorphic_envelop = resample(normalized_homomorphic_envelop, new_length) #new_length
    resampled_homomorphic_envelop = np.expand_dims(resampled_homomorphic_envelop, -1) # (new_length, 1)
    
    
    # **************** Hilbert Envelope *************************
    hilbert_envelop = hilbert_env(audio= denoise_audio)
    # plot_signal(hilbert_envelop, 'c')
    normalized_hilbert_envelop = normalize(x= hilbert_envelop)
    # plot_signal(normalized_hilbert_envelop, 'd')
    resampled_hilbert_envelop = resample(normalized_hilbert_envelop, new_length)
    resampled_hilbert_envelop = np.expand_dims(resampled_hilbert_envelop, -1)
    
    
    # ********************* PSD Envelope ************************
    PSD_envelope = PSD_env(audio= audio, sr= sampling_rate)
    # plot_signal(PSD_envelope, 'e')
    normalized_PSD_envelope = normalize(x= PSD_envelope)
    # plot_signal(normalized_PSD_envelope, 'f')
    resampled_PSD_envelope = resample(normalized_PSD_envelope, new_length)
    resampled_PSD_envelope = np.expand_dims(resampled_PSD_envelope, -1)

    
    audio_info = np.concatenate((audio_info, resampled_homomorphic_envelop), axis= 1)
    audio_info = np.concatenate((audio_info, resampled_hilbert_envelop), axis= 1)
    audio_info = np.concatenate((audio_info, resampled_PSD_envelope), axis= 1)
    
    
    # ********************** Wavelets Decomposition ********************
    wav_coeffs = \
        stationary_wavelets_decomposition(audio, wavelet= 'rbio3.9', 
                                          levels=[0,1,2,3,4],
                                          start_level=int(np.log2(sampling_rate//1000)), 
                                          end_level=int(np.log2(sampling_rate//20)), 
                                          erase_pad=True)
    for i in range(wav_coeffs.shape[1]):
        try:
            wavelet_norm = homorphic_env(wav_coeffs[:,i], sampling_rate)

            # Concatenando
            wavelet_norm = np.expand_dims(resample(wavelet_norm,new_length), -1)
            # print(wavelet_norm.shape)
            audio_info = np.concatenate((audio_info, wavelet_norm), axis=1)
        except:
            print("Catch")
    
    mfcc_feature = get_mfcc_features(audio= denoise_audio, sampling_rate= sampling_rate, new_length= new_length)
    delta_mfcc = get_delta_mfcc(mfcc_coeff= mfcc_feature, new_length= new_length)
    delta_delta_mfcc = get_delta_mfcc(mfcc_coeff= delta_mfcc, new_length= new_length)
    
    audio_info = np.concatenate((audio_info, mfcc_feature), axis=1)
    audio_info = np.concatenate((audio_info, delta_mfcc), axis=1)
    audio_info = np.concatenate((audio_info, delta_delta_mfcc), axis=1)
    
    # print(audio_info.shape)
    return audio_info


def get_window_signal(signal_in, N= 512, noverlap= 0, padding_value = 2):
    noverlap = int(noverlap)
    signal_out = list()
    step = N - noverlap
    
    while signal_in.shape[0] != 0:
        if signal_in.shape[0] >= N:
            signal_frame = signal_in[:N]
            signal_in = signal_in[step:]
        else:
            last_frame = signal_in[:]
            if signal_in.ndim == 1:
                signal_frame = np.zeros(N) + padding_value
                signal_frame[:last_frame.shape[0]] = last_frame
            else:
                signal_frame = np.zeros((N, last_frame.shape[1])) + \
                               padding_value
                signal_frame[:last_frame.shape[0], 
                             :last_frame.shape[1]] = last_frame
            signal_in = signal_in[:0]
        signal_out.append(signal_frame)
    return np.array(signal_out)

def get_original_signal(probablity_output, envelope, N, noverlap, padding_value = 2):
    noverlap = int(noverlap)
    signal_out = np.zeros((probablity_output.shape[0] * probablity_output.shape[1] , 4))
    step = N - noverlap
    for i in range(0, len(probablity_output)):
        signal_out[i*step:(i*step + N), :] = probablity_output[i, :, :]
    return np.array(signal_out[:len(envelope), :])
    
def run_model(data):
    model = getModel()
    predictions = model.predict(data)
    return np.array(predictions)

# def rearrange_prob(probablities):
#     modified_predictions = np.zeros((probablities.shape[0], 4))
#     modified_predictions[:, 0] = probablities[:, 1]  # S1
#     modified_predictions[:, 1] = probablities[:, 0]  # Other
#     modified_predictions[:, 2] = probablities[:, 2]  # S2
#     modified_predictions[:, 3] = probablities[:, 0]  # Other
#     return np.array(modified_predictions)

def get_features(audio_file):
    slice_length,downsample_rate,label_sampling_rate,noverlap,padding_value,append_envelopes = set_default_parameter()
    window_length = slice_length * label_sampling_rate
    noverlap_length = noverlap * window_length
    # print("Window Length and NOverlap Length", window_length, noverlap_length)
    signal, sampling_rate = librosa.load(audio_file, sr=downsample_rate)
    # print(len(signal) / sampling_rate)
    signal = signal[0: len(signal)//2 * 2]
    envelop = get_audio_features(audio= signal, sampling_rate= sampling_rate)
    # print("Envelope Shape", envelop.shape)
    data = get_window_signal(signal_in= envelop, N= window_length, noverlap= noverlap_length, padding_value= 0)
    # print("Windowed Shape", data.shape)
    probablity = run_model(data= data)
    posteriors = get_original_signal(probablity_output= probablity, envelope= envelop, N= window_length, noverlap= noverlap_length, padding_value= 0)
    # posteriors = rearrange_prob(probablities= posteriors)
    
    return np.array(posteriors)