from scipy import signal
import librosa
import numpy as np
import pandas as pd



def norm(arr):
    return ((arr - np.min(arr))/np.ptp(arr)).astype(np.float16)


def shrink(arr):
    #gaussian blur kernel
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump)
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    return signal.fftconvolve(arr, kernel, mode='same')


def extract(file):
    hop_length = 512
    x, sr = librosa.load(file)
    X = librosa.stft(x)

    Xdb = librosa.amplitude_to_db(abs(X))
    zcrs = librosa.feature.zero_crossing_rate(x)
    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)


    oenv = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                          hop_length=hop_length)

    S, phase = librosa.magphase(librosa.stft(x))
    rms = librosa.feature.rms(S=S)

    vls = [Xdb,rms,zcrs,mfccs,chromagram,tempogram]

    FD = []
    for a in vls:
        FD = norm(a) if len(FD) == 0 else np.concatenate((FD,norm(a)),0)
    FD = np.reshape(shrink(FD),[FD.shape[1],FD.shape[0]])

    pd.DataFrame(FD).to_csv(f"{file}_{FD.shape[0]}_{FD.shape[1]}.csv",header=False, index=False)