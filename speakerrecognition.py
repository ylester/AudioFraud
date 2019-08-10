# Import classifier model
from sklearn.ensemble import RandomForestClassifier

# Feature manipulation
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

# Import data
from audio_data import get_data


def clean_data(df):
    encoder = LabelEncoder()
    df["person"] = encoder.fit_transform(df.person)
    return df


def make_pickle(name, obj):
    pickle_out = open(name, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def extract_features(df, wanted):
    column_names = list(df)
    y_feature = column_names.index(wanted[-1])
    X_features = [column_names.index(x) for x in wanted[:-1]]
    return [X_features, y_feature]


def train_model(df, features):
    X_features = features[0:-1]
    y_feature = features[-1]
    model = RandomForestClassifier()
    X = df[X_features].values
    y = df[y_feature].values.ravel()
    model.fit(X, y)
    return model


def get_speaker(speaker):
    speakers = {0: "Sedrick",
                1: "Shawn",
                2: "Unknown",
                3:"Yesha"}

    return speakers[speaker]


def identify_speaker(input):
    model = pd.read_pickle("data/classifiers/voice_model")
    features = [str(i) for i in range(1,102)] + ["fbankmean", "mfcc_mean"]
    input = input[features]
    result = model.predict(input)
    speaker = get_speaker(result)
    return speaker


def main():
    df = get_data()
    df = clean_data(df)
    features = [str(i) for i in range(1,102)] + ["fbankmean", "mfcc_mean", "person"]
    model = train_model(df, features)
    name = "data/classifiers/voice_model"
    make_pickle(name, model)


main()


