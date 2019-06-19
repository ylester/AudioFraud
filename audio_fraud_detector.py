import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, auc, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels


def create_dataframe():
	path = "ClonedSamples"
	files = os.listdir(path)
	df = pd.DataFrame()
	cg_audio = []
	cg_mfcc = []
	cg_filter_bank = []
	cg_rates = []

	for audio in files:
		if ".wav" in audio:
			cg_audio.append(audio)
			rate, sig = wav.read(audio)
			cg_rates.append(rate/1000)
			mfcc_feature = mfcc(sig, rate)
			cg_mfcc.append(mfcc_feature)
			fbank_feat = logfbank(sig, rate)
			cg_filter_bank.append(fbank_feat)

	df['computer_generated_audio'] = cg_audio
	df['rates'] = cg_rates
	df['mfcc'] = cg_mfcc
	df['filter_bank'] = cg_filter_bank
	return df


def data_visualization(df):
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
	pass


def extract_features(*kwargs):
    pass


def analyze_data(*kwargs):
	pass


def detect_fraud(*kwargs):
	pass


def send_results_to_hardware(*kwargs):
	pass


if __name__ == "__main__":
	data = create_dataframe()
	data_visualization(data)