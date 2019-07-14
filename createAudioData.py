import os
import scipy.io.wavfile as wav
import pandas as pd
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


def get_filename(file):
    return os.fsdecode(file)


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
    z = []
    mfccs = []
    filter_bank = []

    counter = 1
    for index, person in enumerate(os.listdir(data_dir)):
        if person in ["authentic", "original", "fraud"]:
            continue
        person_dir = get_person_dir(person, data_dir)
        person_dir = os.path.realpath(person_dir)
        files = os.listdir(os.fsencode(person_dir))
        for audio_file in files:
            if isinstance(audio_file, str):
                print(audio_file)
                print(counter)
                break
            audio_file = audio_file.decode("utf-8")
            audio_file = person_dir + "/" + audio_file
            rate, stereo = wav.read(audio_file)
            files.append(audio_file)
            people.append(person)
            rates.append(rate)
            speaker_nums.append(index + 1)
            mono = stereo[:,0]
            f, t, Zxx = stft(mono, rate, nperseg=200)
            fbank_feat = logfbank(stereo, rate, nfft=1103)
            mfcc_feature = mfcc(stereo, rate, nfft=1103)
            filter_bank.append(fbank_feat)
            mfccs.append(mfcc_feature)
            freq.append(f)
            z.append(Zxx.T)
            filename = get_filename(audio_file)
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
            counter += 1

    df['file'] = files
    df["person"] = people
    df['rate'] = rates
    df["speaker_num"] = speaker_nums
    df["frequency"] = freq
    df["voiceprint"] = z
    df["voiceprint"] = df["voiceprint"].abs().mean()
    df["filename"] = file_names
    df["fraud"] = fraud
    df["authentic"] = authentic
    df["recorded"] = recorded
    df["computer_generated"] = cg
    df['filter_bank'] = filter_bank
    df['mfcc'] = mfccs

    df = df.reindex(sorted(df.columns), axis=1)

    return df

data_dir = "data"
# create_data(dir)
df = create_dataframe(data_dir)
filename = "/audio_data.csv"
path = os.path.join(data_dir)
path = os.path.realpath + filename
df.to_csv(path)