import os
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
from python_speech_features import mfcc, logfbank
from pydub import AudioSegment
from scipy.signal import stft
import glob


def create_data(data_dir):
    recorded_dir = os.path.join(data_dir, "fraud/recorded")
    recorded_dir = os.path.realpath(recorded_dir)
    auth_dir = os.path.join(data_dir, "authentic")
    auth_dir = os.path.realpath(auth_dir)
    original_dir = os.path.join(data_dir, "original")
    data_dir = os.path.realpath(data_dir)
    path = os.path.realpath(original_dir)
    files = os.listdir(os.fsencode(path))
    # iterate over wav files
    for file in files:
        file = file.decode("utf-8")
        filename = get_filename(file)
        file = path + "/" + file
        # full numbers in the 1000s means seconds
        if filename.endswith("wav"):
            start = 0
            end = 6000
            num_index = get_num_index(filename)
            dot_index = get_dot_index(filename)
            person = filename[:num_index]
            new_filename = filename[:dot_index]
            file = open(file, "rb")
            sound = AudioSegment.from_wav(file)
            while end <= len(sound):
                new_file = sound[start:end]
                new_filename += str(int((start/1000))) + "-" + str(int((end/1000))) + "seconds.wav"
                new_file.export(data_dir + "/" + person + "/" + new_filename, format="wav")
                if is_fraud(filename)[0]:
                    new_file.export(recorded_dir + "/" + new_filename, format="wav")
                else:
                    new_file.export(auth_dir + "/" + new_filename, format="wav")
                start = end
                end += 6000
                new_filename = filename[:dot_index]
            file.close()


def get_num_index(string):
    for i in range(len(string)):
        if string[i].isdigit():
            return i
    return -1


def get_dot_index(string):
    return string.index(".")


def get_person_dir(person, data_dir):
    return os.path.join(data_dir,person)


def is_fraud(filename):
    if "recorded" in filename:
        return True, 1
    elif "cg" in filename:
        return True, 2
    else:
        return False, 0


def get_filename(file, person):
    lastchar_index = file.index(person[-1]) + 2
    return file[lastchar_index:]


def create_dataframe(data_dir):
    df = pd.DataFrame()
    files = []
    file_names = []
    authentic = []
    fraud = []
    recorded = []
    cg = []
    people = []
    rates = []
    speaker_nums = []
    freq = []
    zmean = []
    z = []
    mfccs = []
    filter_bank = []
    mono = None

    for index, person in enumerate(os.listdir(data_dir)):
        if person in ["authentic", "original", "fraud", "shawn",  'audio_data.xlsx', 'audio_data.csv']:
            continue
        person_dir = get_person_dir(person, data_dir)
        path = person_dir + "/*.wav"
        for audio_file in glob.glob(path):
            print(audio_file)
            rate, stereo = wav.read(audio_file)
            if len(stereo) == 0:
                continue
            mono = stereo[:, 0]
            files.append(audio_file)
            people.append(person)
            rates.append(rate)
            speaker_nums.append(index + 1)
            f, t, Zxx = stft(mono, rate, nperseg=200)
            fbank_feat = logfbank(stereo, rate, nfft=1103)
            mfcc_feature = mfcc(stereo, rate, nfft=1103)
            filter_bank.append(fbank_feat)
            mfccs.append(mfcc_feature)
            freq.append(f)
            new_zxx = pd.DataFrame(Zxx.T).abs().mean().values
            z.append(Zxx.T)
            zmean.append(new_zxx)
            filename = get_filename(audio_file, person)
            file_names.append(filename)
            fraud_value, fraud_type = is_fraud(filename)
            if fraud_value:
                if fraud_type == 1:
                    recorded.append(1)
                    cg.append(0)
                else:
                    recorded.append(0)
                    cg.append(1)
                fraud.append(1)
                authentic.append(0)
            else:
                authentic.append(1)
                fraud.append(0)
                cg.append(0)
                recorded.append(0)

    df['file'] = files
    df["person"] = people
    df['rate'] = rates
    df["speaker_num"] = speaker_nums
    df["frequency"] = freq
    df["voiceprint_mean"] = zmean
    df["voiceprint"] = z
    df["filename"] = file_names
    df["fraud"] = fraud
    df["authentic"] = authentic
    df["recorded"] = recorded
    df["computer_generated"] = cg
    df['filter_bank'] = filter_bank
    df['mfcc'] = mfccs

    df = df.reindex(sorted(df.columns), axis=1)

    return df


def get_data():
    excel = "data/audio_data.xlsx"
    df = pd.read_excel(excel)
    # df = clean_data(df)
    return df

def clean_data(df):
    cols = ["voiceprint", "filter_bank", "mfcc"]
    for col in cols:
        df[col] = df[col].apply(convertStringToArray, col)
    return df



def create_csv(df, filename):
    df.to_csv(filename)

def create_excel(df, filename):
    df.to_excel(filename)


def find_parens(s):
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '[':
            pstack.append(i)
        elif c == ']':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret
def convertStringToArray(string):
    result = []
    print(string)
    array = string[1:-1]
    parens = find_parens(array)
    for open_paren in sorted(parens.keys()):
        start = open_paren + 1
        end = parens[open_paren]
        subarray = array[start:end]
        subarray = subarray.split(" ")
        subarray = [float(x) for x in subarray]
        result.append(subarray)
    return result

# data_dir = "data"
# df = create_dataframe(data_dir)
# filename = "data/audio_data.csv"
# create_csv(df, filename)
# filename = "data/audio_data.xlsx"
# create_excel(df, filename)


