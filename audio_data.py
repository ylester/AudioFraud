import glob, os, pickle
import scipy.io.wavfile as wavf
import pandas as pd
import numpy as np
from python_speech_features import mfcc, logfbank
from pydub import AudioSegment
from scipy.signal import stft


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
    if person == "yesha":
        lastchar_index = file.index(person) + len(person) - 1 + 2
    else:
        lastchar_index = file.index(person[-1]) + 2
    return file[lastchar_index:]

def make_pickle(name, object):
    pickle_out = open(name, "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()
    return pickle_out


def create_dataframe(data_dir):
    rows = []

    mono = None

    for index, person in enumerate(os.listdir(data_dir)):
        if person in ["authentic", "original", "fraud", "audio_data.csv"]:
            continue
        person_dir = get_person_dir(person, data_dir)
        path = person_dir + "/*.wav"
        for audio_file in glob.glob(path):
            row = {}

            print(audio_file)
            filename = get_filename(audio_file, person)
            row["filename"] = filename
            row["file"] = audio_file
            row["person"] = person
            row["speaker_num"] = index

            audio_file = open(audio_file, "rb")

            rate, stereo = wavf.read(audio_file)
            if not isinstance(stereo[0], np.ndarray):
                mono = stereo
            else:
                mono = stereo[:, 0]


            row["rate"] = rate

            f, t, Zxx = stft(mono, rate, nperseg=200)
            temp = pd.DataFrame(Zxx.T).abs().mean().values
            count = 1
            for val in temp:
                row[count] = val
                count += 1

            fbank_mean = np.mean(logfbank(stereo, rate, nfft=1200))
            row["fbankmean"] = fbank_mean

            mfcc_mean = np.mean(mfcc(stereo, rate, nfft=1200))
            row["mfcc_mean"] = mfcc_mean

            fraud_value, fraud_type = is_fraud(filename)
            if fraud_value:
                if fraud_type == 1:
                    row["recorded"] = 1
                    row["computer_generated"] = 0
                else:
                    row["recorded"] = 0
                    row["computer_generated"] = 1
                row["fraud"] = 1
                row["authentic"] = 0
            else:
                row["authentic"] = 1
                row["fraud"] = 0
                row["computer_generated"] = 0
                row["recorded"] = 0

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_data():
    csv = "data/audio_data.csv"
    df = pd.read_csv(csv)
    return df


def create_csv(df, filename):
    df.to_csv(filename)





