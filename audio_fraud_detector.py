import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sklearn as sk
from io import StringIO
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def create_cga_dataframe():
    """
    This module creates a computer generated audio data frame
    by extracting features from the audio files and storing
    the data
    """

    cg_path = "ClonedSamples/*.wav"
    cg_audio = []
    cg_mfcc = [] # Cepstrum is the information of rate of change in spectral bands
    cg_filter_bank = []
    cg_rates = []
    cg_fraud = []
    cg_auth = []
    testtry = []
    scaler = sk.preprocessing.StandardScaler()

    df = pd.DataFrame()

    for wave_file in glob.glob(cg_path):
        cg_audio.append(wave_file)
        rate, sig = wav.read(wave_file)
        cg_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        cg_mfcc.append(mfcc_feature)
        mfcctest = sk.preprocessing.scale(mfcc_feature, axis=1)
        testtry.append(mfcctest)
        # print(type(mfcc_feature))
        mfcctest_scaled = scaler.fit_transform(mfcctest)
        mfccscaled = scaler.fit_transform(mfcc_feature)
        # print(mfccscaled)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        cg_filter_bank.append(fbank_feat)
        cg_fraud.append(1)
        cg_auth.append(0)


    # df['computer_generated_audio'] = cg_audio
    df['rates'] = cg_rates
    df['mfcc'] = np.array(cg_mfcc).flatten()
    # df['computer_generated_mfcc'] = df['computer_generated_mfcc'].astype(object)
    df['filter_bank'] = np.array(cg_filter_bank).flatten()
    # df['computer_generated_filter_bank']= df['computer_generated_filter_bank'].astype(object)
    df['fraud'] = cg_fraud
    df['authentic'] = cg_auth

    return df


def create_aa_dataframe():
    """
    This module creates an authentic audio data frame by extracting features
    from the audio files and storing the data
    """

    auth_sed = "og_data/sedrick/*.wav"
    auth_esh = "og_data/yesha/*.wav"
    auth_audio = []
    auth_mfcc = []
    auth_filter_bank = []
    auth_rates = []
    auth_fraud = []
    auth_auth = []
    df2 = pd.DataFrame()

    for wave_file in glob.glob(auth_esh):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        auth_audio.append(wave_file)
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        auth_mfcc.append(np.array(mfcc_feature, dtype=int))
        fbank_feat = logfbank(sig, rate, nfft=1103)
        auth_filter_bank.append(fbank_feat)
        auth_fraud.append(0)
        auth_auth.append(1)

    for wave_file in glob.glob(auth_sed):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        auth_audio.append(wave_file)
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        auth_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1103)
        auth_filter_bank.append(fbank_feat)
        auth_fraud.append(0)
        auth_auth.append(1)

    # df2['authentic_audio'] = auth_audio
    df2['rates'] = auth_rates
    df2['mfcc'] = auth_mfcc
    df2['filter_bank'] = auth_filter_bank
    df2['fraud'] = auth_fraud
    df2['authentic'] = auth_auth

    csv_loc = "authentic.csv"
    df2.to_csv(csv_loc)

    # return df2


def analyze_computer_generated_audio_data(df):
    """
    This module is to visualize and analyze the data and features
    """

    # Analyze audio rate data
    rates = df['computer_generated_rates'].value_counts()
    labels = ["16kHz", "48kHz"]
    plt.pie(rates, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Computer Generated Audio Rate Data")
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


def analyze_authentic_audio_data(df):
    """
    This module is to visualize and analyze the data and features
    """

    # Analyze audio rate data
    rates = df['authentic_rates'].value_counts()
    labels = ["44.1kHz"]
    plt.pie(rates, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Authentic Audio Rate Data")
    plt.show()

    # Analyze the mel-frequency data
    mfcc = df.authentic_mfcc
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
    plt.plot(mfcc_visual[5])
    plt.plot(mfcc_visual[10])
    plt.title("MFCC Features (3)")
    plt.xlabel("Features (Per Color)")
    plt.ylabel("Frequency")
    plt.show()

    # Analyze filter bank data
    fbank = df.authentic_filter_bank
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
    sampling_frequency, signal_data = wav.read(df['authentic_audio'][0])
    plt.subplot(211)
    plt.title('Spectrogram of an authentic audio wav file')
    plt.plot(signal_data)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.specgram(signal_data[:, 0], Fs=sampling_frequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def detect_fraud(cg_df, auth_df, features, target, input_audio=None):
    """
    This module will train the ML module with df inputs to detect whether
    an input audio is fraudulent or non-fraudulent
    """
    df = pd.concat([cg_df, auth_df], sort=False)
    print(df['fraud'].values)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train.values.ravel())
    y_pred = classifier.predict(X_test)
    print(y_pred)

    # Calculate Confusion Matrix for FRAUD audio
    cm_fraud = confusion_matrix(X_test, y_pred)
    print("\nFraudulent Audio Confusion Matrix: ", cm_fraud)

    # # Calculate Confusion Matrix for AUTHENTIC audio
    # cm_auth = confusion_matrix(y_test, y_pred)
    # print("\nNon-Fraudulent Audio Confusion Matrix: ", cm_auth)
    #
    # # Compute Area Under the Receiver Operating Characteristic Curve (FRAUD)
    # auc_fraud = roc_auc_score(X_test, y_pred)
    # print("\nROC AUC (FRAUD): ", auc_fraud)
    #
    # # Compute Area Under the Receiver Operating Characteristic Curve (AUTHENTUC)
    # auc_auth = roc_auc_score(y_test, y_pred_AUTH)
    # print("\nROC AUC (AUTHENTIC): ", auc_auth)
    #
    # # Compute F-Score (FRAUD)
    # fscore_fraud = f1_score(X_test, y_pred_FRAUD)
    # print("\nF-Score Fraud: ", fscore_fraud)
    #
    # # Compute F-Score (AUTHENTIC)
    # fscore_auth = f1_score(y_test, y_pred_AUTH)
    # print("\nF-Score Authentic: ", fscore_auth)

    # Logic for input audio coming soon


def import_audio_data(*kwargs):
    # Need to import Shawn's Module
    # that communicates with the wifi module
    pass


def send_results_to_hardware(*kwargs):
    # Need to import Shawn's Module 
    # that communicates with the wifi module
    pass


if __name__ == "__main__":
    computer_generated_audio_data = create_cga_dataframe()[:126]
    # analyze_computer_generated_audio_data(computer_generated_audio_data)
    # create_aa_dataframe()
    authentic_audio_data = pd.read_csv("authentic.csv")
    # print(authentic_audio_data.head())
    features = ['rates', 'authentic']
    target = ['fraud']

    # print(authentic_audio_data['authentic_mfcc'].head())
    # print(computer_generated_audio_data['computer_generated_mfcc'].head())
    # print(authentic_audio_data.head())
    # for val in authentic_audio_data['authentic_mfcc']:
    #     print(val)
        # test = np.fromstring(val, sep=',')
        # print(test)
    # print(authentic_audio_data.head())
    # print([type(int(x)) for x in authentic_audio_data['authentic_mfcc']])
    # analyze_authentic_audio_data(authentic_audio_data)
    # print(computer_generated_audio_data.size)
    # print(computer_generated_audio_data.shape, authentic_audio_data.shape)
    detect_fraud(computer_generated_audio_data, authentic_audio_data, features, target)
