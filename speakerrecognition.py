# Import classifier model
from sklearn.ensemble import RandomForestClassifier

# Feature manipulation
import pandas as pd


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
    model = pd.read_pickle("data/classifiers/voice_model1")
    features = [str(i) for i in range(1,102)] + ["fbankmean", "mfcc_mean"]
    input = input[features]
    result = model.predict(input)
    speaker = get_speaker(result)
    return speaker



