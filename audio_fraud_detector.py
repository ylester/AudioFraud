import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sklearn as sk
import random
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve


def create_cga_dataframe():
    """
    This module creates a computer generated audio data frame
    by extracting features from the audio files and storing
    the data
    """

    cg_path = "ClonedSamples/*.wav"
    cg_mfcc = [] # Cepstrum is the information of rate of change in spectral bands
    cg_filter_bank = []
    cg_rates = []
    cg_fraud = []
    cg_mfcc_means = []
    cg_fbank_means = []
    df = pd.DataFrame()

    for wave_file in glob.glob(cg_path):
        rate, sig = wav.read(wave_file)
        cg_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        cg_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        cg_filter_bank.append(fbank_feat)
        cg_fraud.append(1)

    for i in range(len(cg_mfcc)):
        cg_mfcc_means.append(np.mean(cg_mfcc[i]))
        cg_fbank_means.append(np.mean(cg_filter_bank[i]))

    df['rates'] = cg_rates
    df['mfcc'] = np.array(cg_mfcc).flatten()
    df['filter_bank'] = np.array(cg_filter_bank).flatten()
    df['fraud'] = cg_fraud
    df['mfcc_mean'] = cg_mfcc_means
    df['fbank_mean'] = cg_fbank_means
    csv_loc = "fraud.csv"
    df.to_csv(csv_loc)


def create_aa_dataframe():
    """
    This module creates an authentic audio data frame by extracting features
    from the audio files and storing the data
    """

    auth_sed = "og_data/sedrick/*.wav"
    auth_esh = "og_data/yesha/*.wav"
    auth_mfcc = []
    auth_filter_bank = []
    auth_rates = []
    auth_fraud = []
    auth_mfcc_means = []
    auth_fbank_means = []
    df2 = pd.DataFrame()

    for wave_file in glob.glob(auth_esh):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        auth_mfcc.append(np.array(mfcc_feature).flatten())
        fbank_feat = logfbank(sig, rate, nfft=1103)
        auth_filter_bank.append(np.array(fbank_feat).flatten())
        auth_fraud.append(0)

    for wave_file in glob.glob(auth_sed):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        auth_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1103)
        auth_filter_bank.append(fbank_feat)
        auth_fraud.append(0)

    for i in range(len(auth_mfcc)):
        auth_mfcc_means.append(np.mean(auth_mfcc[i]))
        auth_fbank_means.append(np.mean(auth_filter_bank[i]))

    df2['rates'] = auth_rates
    df2['mfcc'] = auth_mfcc
    df2['filter_bank'] = auth_filter_bank
    df2['fraud'] = auth_fraud
    df2['mfcc_mean'] = auth_mfcc_means
    df2['fbank_mean'] = auth_fbank_means
    csv_loc = "authentic.csv"
    df2.to_csv(csv_loc)


def create_ra_dataframe():
    """
        This module creates a computer generated audio data frame
        by extracting features from the audio files and storing
        the data
        """

    r_path = "data/fraud/recorded/*.wav"
    r_mfcc = []  # Cepstrum is the information of rate of change in spectral bands
    r_filter_bank = []
    r_rates = []
    r_fraud = []
    r_mfcc_means = []
    r_fbank_means = []

    df = pd.DataFrame()

    for wave_file in glob.glob(r_path):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        r_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1103)
        r_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1103)
        r_filter_bank.append(fbank_feat)
        r_mfcc_means.append(np.mean(mfcc_feature))
        r_fbank_means.append(np.mean(fbank_feat))
        r_fraud.append(1)


    df['rates'] = r_rates
    df['mfcc'] = r_mfcc
    df['filter_bank'] = r_filter_bank
    df['fraud'] = r_fraud
    df['mfcc_mean'] = r_mfcc_means
    df['fbank_mean'] = r_fbank_means
    csv_loc = "recorded.csv"
    df.to_csv(csv_loc)


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


def train_model(cg_df, auth_df, rec_df, features, target):
    """
    This module will train the ML module with df inputs to detect whether
    an input audio is fraudulent or non-fraudulent
    """
    df = pd.concat([cg_df, auth_df, rec_df], sort=False)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=None)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate Confusion Matrix for FRAUD audio
    cm_fraud = confusion_matrix(y_test, y_pred)
    print "Fraudulent Audio Confusion Matrix:", cm_fraud

    # Compute Area Under the Receiver Operating Characteristic Curve (FRAUD)
    auc_fraud = roc_auc_score(y_test, y_pred)
    print "ROC AUC (FRAUD): ", auc_fraud

    # Compute F-Score (FRAUD)
    fscore_fraud = f1_score(y_test, y_pred)
    print "F-Score Fraud: ", fscore_fraud

    # Compute Precision Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    print "Precision:", precision
    print "Recall:", recall
    print "Threshold:", thresholds
    pr_auc = sk.metrics.auc(recall, precision)
    fpr, tpr, threshold = sk.metrics.roc_curve(y_test, y_pred)
    roc_auc = sk.metrics.auc(fpr, tpr)
    print "PR_AUC:", pr_auc

    # # Training Plots
    # plt.title("Precision-Recall vs Threshold Chart")
    # plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    # plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    # plt.ylabel("Precision, Recall")
    # plt.xlabel("Threshold")
    # plt.legend(loc="lower left")
    # plt.ylim([0, 1])
    # plt.show()
    #
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    #
    # # Classifications based on feature pairs
    # plt.title('MFCC Mean VS Fbank Mean')
    # plt.scatter(cg_df['mfcc_mean'], cg_df['fbank_mean'], color='blue', label='Computer Generated')
    # plt.scatter(auth_df['mfcc_mean'], auth_df['fbank_mean'], color='red', label='Authentic')
    # plt.scatter(rec_df['mfcc_mean'], rec_df['fbank_mean'], color='green', label='Recorded')
    # plt.ylabel('Filter Bank')
    # plt.xlabel('MFCC')
    # plt.legend()
    # plt.show()
    #
    # plt.title('MFCC Mean VS Rates')
    # plt.scatter(cg_df['mfcc_mean'], cg_df['rates'], color='blue', label='Computer Generated')
    # plt.scatter(auth_df['mfcc_mean'], auth_df['rates'], color='red', label='Authentic')
    # plt.scatter(rec_df['mfcc_mean'], rec_df['rates'], color='green', label='Recorded')
    # plt.ylabel('Rates')
    # plt.xlabel('MFCC')
    # plt.legend()
    # plt.show()
    #
    # plt.title('Rates VS Filter Bank')
    # plt.scatter(cg_df['rates'], cg_df['fbank_mean'], color='blue', label='Computer Generated')
    # plt.scatter(auth_df['rates'], auth_df['fbank_mean'], color='red', label='Authentic')
    # plt.scatter(rec_df['rates'], rec_df['fbank_mean'], color='green', label='Recorded')
    # plt.ylabel('Filter Bank')
    # plt.xlabel('Rates')
    # plt.legend()
    # plt.show()

    print
    return classifier


def detect_fraud(clf, input_audio=None):
    result = clf.predict(input_audio)
    if result == 1:
        print("Input Audio Marked as [FRAUD]")
    else:
        print("Input Audio Marked as [AUTHENTIC]")
    np.savetxt("output.txt", result)


def extract_input_audio_features(audio):
    df = pd.DataFrame()
    rate, sig = wav.read(audio)
    if len(sig) == 0:
        return 0
    mfcc_feature = mfcc(sig, rate, nfft=1200)
    fbank_feat = logfbank(sig, rate, nfft=1200)

    df['rates'] = [rate]
    df['mfcc_mean'] = np.mean(mfcc_feature)
    df['fbank_mean'] = np.mean(fbank_feat)
    return df


def import_audio_data(*kwargs):
    # Need to import Shawn's Module
    pass


def send_results_to_hardware(*kwargs):
    # Need to import Shawn's Module
    pass


def plot_result(data, cg_df, auth_df, rec_df):
    plt.title('Classification For Input Data')
    plt.scatter(cg_df['mfcc_mean'], cg_df['fbank_mean'], color='blue', label='Computer Generated')
    plt.scatter(auth_df['mfcc_mean'], auth_df['fbank_mean'], color='red', label='Authentic')
    plt.scatter(rec_df['mfcc_mean'], rec_df['fbank_mean'], color='green', label='Recorded')
    plt.scatter(data['mfcc_mean'], data['fbank_mean'], color='black', label='Predicted')
    plt.ylabel('Filter Bank')
    plt.xlabel('MFCC')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cga_data = pd.read_csv("fraud.csv").sample(126, random_state=1)
    ra_data = pd.read_csv("recorded.csv").sample(126, random_state=1)
    aa_data = pd.read_csv("authentic.csv").sample(126, random_state=1)
    features = ['rates', 'mfcc_mean', 'fbank_mean']
    target = ['fraud']

    auth_path = "data/authentic/*.wav"
    fraud_path = "data/fraud/recorded/*.wav"
    random_fraud_audio = random.choice(glob.glob(fraud_path))
    random_auth_audio = random.choice(glob.glob(auth_path))

    example_audio = extract_input_audio_features(random_auth_audio)
    plot_result(example_audio, cga_data, aa_data, ra_data)
    clf = train_model(cga_data, aa_data, ra_data, features, target)
    detect_fraud(clf, example_audio)
