import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sklearn as sk
import random, os
import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve


def create_cga_dataframe():
    """
    This module creates a computer generated audio data frame
    by extracting features from the audio files and storing
    the data
    """

    cg_path = "data/computer_generated/*/*.wav"
    cg_mfcc = [] # Cepstrum is the information of rate of change in spectral bands
    cg_filter_bank = []
    cg_rates = []
    cg_fraud = []
    cg_mfcc_means = []
    cg_fbank_means = []
    df = pd.DataFrame()

    for wave_file in glob.glob(cg_path):
        print(wave_file)
        rate, sig = wav.read(wave_file)
        cg_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        cg_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        cg_filter_bank.append(fbank_feat)
        cg_mfcc_means.append(np.mean(mfcc_feature))
        cg_fbank_means.append(np.mean(fbank_feat))
        cg_fraud.append(1)

    df['rates'] = cg_rates
    df['mfcc'] = cg_mfcc
    df['filter_bank'] = cg_filter_bank
    df['fraud'] = cg_fraud
    df['mfcc_mean'] = cg_mfcc_means
    df['fbank_mean'] = cg_fbank_means
    csv_loc = "data/computer_generated.csv"
    df.to_csv(csv_loc)
    # return df


def create_aa_dataframe():
    """
    This module creates an authentic audio data frame by extracting features
    from the audio files and storing the data
    """

    path = "data/authentic/*.wav"
    auth_mfcc = []
    auth_filter_bank = []
    auth_rates = []
    auth_fraud = []
    auth_mfcc_means = []
    auth_fbank_means = []
    df2 = pd.DataFrame()

    for wave_file in glob.glob(path):
        rate, sig = wav.read(wave_file)
        if len(sig) == 0:
            continue
        auth_rates.append(rate)
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        auth_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
        auth_filter_bank.append(fbank_feat)
        auth_mfcc_means.append(np.mean(mfcc_feature))
        auth_fbank_means.append(np.mean(fbank_feat))
        auth_fraud.append(0)

    df2['rates'] = auth_rates
    df2['mfcc'] = auth_mfcc
    df2['filter_bank'] = auth_filter_bank
    df2['fraud'] = auth_fraud
    df2['mfcc_mean'] = auth_mfcc_means
    df2['fbank_mean'] = auth_fbank_means
    csv_loc = "data/authentic.csv"
    df2.to_csv(csv_loc)


def create_ra_dataframe():
    """
        This module creates a computer generated audio data frame
        by extracting features from the audio files and storing
        the data
        """

    r_path = "data/fraud/recorded/*.wav"
    r_mfcc = []
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
        mfcc_feature = mfcc(sig, rate, nfft=1200)
        r_mfcc.append(mfcc_feature)
        fbank_feat = logfbank(sig, rate, nfft=1200)
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
    csv_loc = "data/recorded.csv"
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
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1)
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train, y_train.values.ravel())
    # y_pred = classifier.predict(X_test)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)

    # Calculate Confusion Matrix for FRAUD audio
    cm_fraud = confusion_matrix(y_test, y_pred)
    print "Fraudulent Audio Confusion Matrix:"
    print cm_fraud
    print

    # Compute Area Under the Receiver Operating Characteristic Curve (FRAUD)
    auc_fraud = roc_auc_score(y_test, y_pred)
    print "ROC AUC: "
    print auc_fraud
    print

    # Compute F-Score (FRAUD)
    fscore_fraud = f1_score(y_test, y_pred)
    print "F-Score: "
    print fscore_fraud
    print

    # Compute Precision Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = sk.metrics.auc(recall, precision)
    fpr, tpr, threshold = sk.metrics.roc_curve(y_test, y_pred)
    roc_auc = sk.metrics.auc(fpr, tpr)
    print "Precision Recall AUC:"
    print pr_auc

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
    return clf


def detect_fraud(clf, input_audio):
    result = clf.predict(input_audio)
    probability = clf.predict_proba(input_audio)
    if result == 1:
        output = ["Input Audio Marked as [FRAUD]", "Certainty: ",
                  "[Not Fraud, Fraud]", probability]
        send_results_to_hardware(output)
        for results in output:
            print results
        return "FRAUD"
    else:
        output = ["Input Audio Marked as [AUTHENTIC]", "Certainty: ",
                  "[Not Fraud, Fraud]", probability]

        send_results_to_hardware(output)
        for results in output:
            print results
        return "AUTHENTIC"


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


def send_results_to_hardware(input):
    np.savetxt("output.txt", input, fmt='%s', delimiter=',')


def rename_files():
    path = "data/computer_generated/cg_shawn/"
    for index, filename in enumerate(os.listdir(path)):
        dst = "shawn_cg" + str(index+1) + ".wav"
        src = path + "/" + filename
        dst = path + "/" + dst
        os.rename(src, dst)


def plot_result(data, cg_df, auth_df, rec_df, result):
    plt.title('Classification For Input Data')
    plt.scatter(cg_df['mfcc_mean'], cg_df['fbank_mean'], color='blue', label='Computer Generated')
    plt.scatter(auth_df['mfcc_mean'], auth_df['fbank_mean'], color='red', label='Authentic')
    plt.scatter(rec_df['mfcc_mean'], rec_df['fbank_mean'], color='green', label='Recorded')
    plt.scatter(data['mfcc_mean'], data['fbank_mean'], color='black', label='Input Data Point')
    plt.text(data['mfcc_mean'] + 0.5 , data['fbank_mean'], result, bbox=dict(facecolor='white', alpha=0.5))
    plt.ylabel('Filter Bank')
    plt.xlabel('MFCC')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print
    cga_data = pd.read_csv("data/computer_generated.csv")[:200]
    ra_data = pd.read_csv("data/recorded.csv")[:200]
    aa_data = pd.read_csv("data/authentic.csv")[:200]
    features = ['rates', 'mfcc_mean', 'fbank_mean']
    target = ['fraud']

    # Test To Show Working Model
    random_recorded = pd.read_csv("data/recorded.csv")[features][200:].sample(1)
    random_auth = pd.read_csv("data/authentic.csv")[features][200:].sample(1)
    random_cg = pd.read_csv("data/computer_generated.csv")[features][200:].sample(1)

    example_audio = random_recorded

    # input_audio_path = "data/authentic/*.wav"
    # random_pick = random.choice(glob.glob(input_audio_path))
    # example_audio = extract_input_audio_features(random_pick)

    clf = train_model(cga_data, aa_data, ra_data, features, target)
    output = detect_fraud(clf, example_audio)
    plot_result(example_audio, cga_data, aa_data, ra_data, output)

