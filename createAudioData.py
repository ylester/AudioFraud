# from pydub import AudioSegment
import os
import scipy.io.wavfile as wav
import pandas as pd
from python_speech_features import mfcc, logfbank
from pydub import AudioSegment


exe_directory_in_str = "/Users/sedrick/Documents/AudioFraud/AudioFraud"
data_directory_in_str = "/users/sedrick/Documents/AudioFraud/AudioFraud/data"
directory = os.fsencode(exe_directory_in_str)

def createData():
    #iterate over mp3 files
    print(os.listdir(directory))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #full numbers in the 1000s means seconds
        if filename.endswith("wav"):
            start = 0
            end = 3000
            dotindex = filename.index('.')
            dir_name = filename[:dotindex]
            file = open(file, "rb")
            sound = AudioSegment.from_wav(file)
            while end <= len(sound):
                newfile = sound[start:end]
                newfilename = dir_name + str(int((start/1000))) + "-" + str(int((end/1000))) + "seconds.wav"
                newfile.export(data_directory_in_str + '/' + dir_name + '/' + newfilename, format="wav")
                start = end
                end += 3000
            newfile = sound[end:]
            newfilename = dir_name + "remainingseconds.wav"
            newfile.export(data_directory_in_str + '/' + dir_name + '/' + newfilename,
                           format='wav')
            file.close()
        else:
         continue

def create_dataframe():
    directory = os.getcwd() + "/data"
    df = pd.DataFrame()
    files = []
    filter_bank = []
    mfccs = []
    people = []
    rates = []
    speaker_nums = []

    for index, dir in enumerate(os.listdir(directory)):
        datadir =  directory + "/" + dir
        os.chdir(datadir)
        for audio_file in os.listdir(datadir):
            rate, sig = wav.read(audio_file)
            fbank_feat = logfbank(sig, rate, nfft=1103)
            mfcc_feature = mfcc(sig, rate, nfft=1103)
            files.append(audio_file)
            mfccs.append(mfcc_feature)
            people.append(dir)
            rates.append(rate / 1000)
            speaker_nums.append(index + 1)
            filter_bank.append(fbank_feat)

    df['file'] = files
    df['filter_bank'] = filter_bank
    df['mfcc'] = mfccs
    df["person"] = people
    df['rate'] = rates
    df["speaker_num"] = speaker_nums
    return df

print(create_dataframe())