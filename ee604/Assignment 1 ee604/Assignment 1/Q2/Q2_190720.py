import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    data,sr=librosa.load(audio_path)
    n_fft = 2048 
    hop_length = 512
    spec= librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)
    y=librosa.power_to_db(spec, ref=np.max)
    y=np.abs(y)
    mean = np.mean(y)
    std = np.std(y)
    threshold = mean + 5* std
    if threshold>150:
        class_name="metal"
    else:
        class_name="cardboard"
    
    return class_name
