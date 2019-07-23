##Import classifer models
from sklearn.linear_model import LogisticRegression

#Feature Manuiplaton
from sklearn.model_selection import train_test_split

#Import metrics
from sklearn.metrics import confusion_matrix

#import data
from audio_data import get_data

def clean_data(df):
    df["voiceprint"] = df["voiceprint"].apply(lambda x: eval(x))
    df["speaker_num"] = df["speaker_num"].apply(lambda x: eval(x))
    return df

def extract_features(df, wanted):
    column_names = list(df)
    y_feature = column_names.index(wanted[-1])
    X_features = [column_names.index(x) for x in wanted[:-1]]
    return [X_features, y_feature]


def identify_speaker(df,features):
    X_features = features[0]
    y_features = features[1]
    log_regr = LogisticRegression(random_state=0)
    X = df.iloc[:, X_features].values
    y = df.iloc[:, y_features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    log_regr.fit(X_train, y_train)
    y_pred = log_regr.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def main():
    df = get_data()
    df = clean_data(df)
    features = ["voiceprint", "speaker_num"]
    features = extract_features(df, features)
    identify_speaker(df, features)


main()