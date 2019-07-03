import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io.wavfile as wav
import librosa
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def create_cga_dataframe():
    """
    This module creates a computer generated audio data frame
    by extracting features from the audio files and storing
    the data
    """

    cg_path = "ClonedSamples/*.wav"
    cg_audio = []
    cg_mfcc = []
    cg_filter_bank = []
    cg_rates = []
    df = pd.DataFrame()

    for wave_file in glob.glob(cg_path):
        cg_audio.append(wave_file)
        rate, sig = wav.read(wave_file)
        cg_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        cg_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        cg_filter_bank.append(fbank_feat)

    df['computer_generated_audio'] = cg_audio
    df['computer_generated_rates'] = cg_rates
    df['computer_generated_mfcc'] = cg_mfcc
    df['computer_generated_filter_bank'] = cg_filter_bank

    return df


def create_aa_dataframe():
    """
    This module creates an authentic audio data frame by extracting features
    from the audio files and storing the data
    """

    auth_sed = "data/sedrick/*.wav"
    auth_esh = "data/yesha/*.wav"
    auth_audio = []
    auth_mfcc = []
    auth_filter_bank = []
    auth_rates = []
    df2 = pd.DataFrame()

    for wave_file in glob.glob(auth_esh):
        auth_audio.append(wave_file)
        rate, sig = wav.read(wave_file)
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        auth_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        auth_filter_bank.append(fbank_feat)

    for wave_file in glob.glob(auth_sed):
        auth_audio.append(wave_file)
        rate, sig = wav.read(wave_file)
        auth_rates.append(rate / 1000)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        auth_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        auth_filter_bank.append(fbank_feat)

    df2['authentic_audio'] = auth_audio
    df2['authentic_rates'] = auth_rates
    df2['authentic_mfcc'] = auth_mfcc
    df2['authentic_filter_bank'] = auth_filter_bank

    return df2


def analyze_data(df):
    """
    This module is to visualize and analyze the data and features
    """

    # Analyze audio rate data
    rates = df['computer_generated_rates'].value_counts()
    labels = ["16MHz", "48MHz"]
    plt.pie(rates, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Audio Rate Data (MHz)")
    plt.show()

    # Analyze the mel-frequency data
    mfcc = df.computer_generated_mfcc
    avg_mfcc = []
    for data in mfcc:
        avg_mfcc.append(np.around(np.average(data), decimals=4))
    counts, bins = np.histogram(avg_mfcc)
    plt.hist(bins[:-1], bins, weights=counts, color='c', edgecolor='k')
    plt.axvline(np.asarray(avg_mfcc).mean(), color='k', linestyle='dashed', linewidth=1)
    plt.title("Average MFCC Histogram")
    plt.xlabel("MFCC Feature Vector Average")
    plt.ylabel("# of Avgs")
    plt.text(0, 23, 'Mean: {:.2f}'.format(np.asarray(avg_mfcc).mean()))
    plt.show()
    # Here is showing an example of MFCC features that are contained in ONE audio file
    mfcc_visual = mfcc[0]
    plt.plot(mfcc_visual)
    plt.title("MFCC Features Per Audio File")
    plt.xlabel("Features (Per Color)")
    plt.ylabel("Frequency")
    plt.show()
    # This is to select only 3 features out of the audio and see what that looks like
    plt.plot(mfcc_visual[0])
    plt.plot(mfcc_visual[10])
    plt.plot(mfcc_visual[20])
    plt.title("MFCC Features (3)")
    plt.xlabel("Features (Per Color)")
    plt.ylabel("Frequency")
    plt.show()


    # Analyze filter bank data
    fbank = df.computer_generated_filter_bank
    avg_fbank = []
    for values in fbank:
        avg_fbank.append(np.around(np.average(values), decimals=4))
    plt.bar(np.arange(len(avg_fbank)), avg_fbank)
    plt.title("Filter Bank Averages Per Wave File")
    plt.xlabel("Filter-bank Averages")
    plt.ylabel("Count")
    plt.show()
    counts, bins = np.histogram(avg_fbank)
    plt.hist(bins[:-1], bins, weights=counts, color='b', edgecolor='k')
    plt.title("Filter Bank Frequency Distribution")
    plt.xlabel("Filter-bank Averages")
    plt.ylabel("Count")
    plt.show()
    # Here is showing an example of Filter Bank features that are contained in ONE audio file
    fbank_visual = fbank[1]
    plt.plot(fbank_visual)
    plt.title("Filter Bank Features Per Audio File")
    plt.xlabel("Features (Per Color)")
    plt.ylabel("Frequency")
    plt.show()
    # This is to select only 3 features out of the audio and see what that looks like
    plt.plot(fbank_visual[0])
    plt.plot(fbank_visual[10])
    plt.plot(fbank_visual[20])
    plt.title("Fbank Features (3)")
    plt.xlabel("Features (Per Color)")
    plt.ylabel("Frequency")
    plt.show()

    # Spectrogram snapshot of an Audio Wave File
    sampling_frequency, signal_data = wav.read(df['computer_generated_audio'][0])
    plt.subplot(211)
    plt.title('Spectrogram of an audio wav file')
    plt.plot(signal_data)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.specgram(signal_data, Fs=sampling_frequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def detect_fraud(df):
    train, test = train_test_split(df, test_size=0.2)
    # Will begin training model once I correctly extract
    # all necessary data


def import_audio_data(*kwargs):
    # Need to import Shawn's Python Module
    # that communicates with the wifi module
    pass


def send_results_to_hardware(*kwargs):
    # Need to import Shawn's Python Module
    # that communicates with the wifi module
    pass


if __name__ == "__main__":
    computer_generated_audio_data = create_cga_dataframe()
    # authentic_audio_data = create_aa_dataframe()
    analyze_data(computer_generated_audio_data)

