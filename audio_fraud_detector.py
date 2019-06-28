import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.io.wavfile as wav
from scipy import signal
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def create_dataframe():
    """
    This module creates a dataframe by extracting features
    from the audio files and storing the data in the dataframe
    """

    path = "ClonedSamples/*.wav"
    cg_audio = []
    cg_mfcc = []
    cg_filter_bank = []
    cg_rates = []
    df = pd.DataFrame()

    for wave_file in glob.glob(path):
        cg_audio.append(wave_file)
        rate, sig = wav.read(wave_file)
        cg_rates.append(rate / 1000)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        cg_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        cg_filter_bank.append(fbank_feat)

    df['computer_generated_audio'] = cg_audio
    df['rates'] = cg_rates
    df['mfcc'] = cg_mfcc
    df['filter_bank'] = cg_filter_bank
    # Adding Authentic Audio to Data frame Soon
    return df


def analyze_data(df):
    """This module is to visualize and analyze the data and features"""
    # Analyze audio rate data
    rates = df.rates
    plt.bar(np.arange(len(rates)), rates)
    plt.show()
    # Analyze the mel-frequency data
    mfcc = df.mfcc
    avg_mfcc = []
    for data in mfcc:
        avg_mfcc.append(np.average(data))
    plt.bar(np.arange(len(avg_mfcc)), avg_mfcc)
    plt.show()
    # Analyze filter bank data
    fbank = df.filter_bank
    avg_fbank = []
    for values in fbank:
        avg_fbank.append(np.average(values))
    plt.bar(np.arange(len(avg_fbank)), avg_fbank)
    plt.show()


def import_audio_data(*kwargs):
    # Need to import Shawn's Python Module
    # that communicates with the wifi module
    pass


def detect_fraud(*kwargs):
    pass


def send_results_to_hardware(*kwargs):
    pass


if __name__ == "__main__":
    data = create_dataframe()
    # analyze_data(data)
