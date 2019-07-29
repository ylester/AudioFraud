##Import classifer models
from sklearn.linear_model import LogisticRegression

#Feature Manuiplaton
from sklearn.model_selection import train_test_split

#Import metrics
from sklearn.metrics import confusion_matrix

#import data
from audio_data import get_data

#visual data
import matplotlib as plt
import seaborn as sb
import matplotlib.patches as mpatches

def clean_data(df):
    df["voiceprint_mean"] = df["voiceprint_mean"].apply()
    # df["speaker_num"] = df["speaker_num"].eval()
    return df

def extract_features(df, wanted):
    column_names = list(df)
    y_feature = column_names.index(wanted[-1])
    X_features = [column_names.index(x) for x in wanted[:-1]]
    return [X_features, y_feature]


def identify_speaker(df,features):
    X_features = features[0:-1]
    y_feature = features[-1]
    log_regr = LogisticRegression(random_state=0)
    X = df[X_features].values
    y = df[y_feature].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    log_regr.fit(X_train, y_train)
    y_pred = log_regr.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.scatter(X, 101, c=y, edgecolor='black')
    patch0 = mpatches.Patch(color='#FF0000', label='yesha')
    patch1 = mpatches.Patch(color='#00FF00', label='sedrick')
    plt.legend(handles=[patch0, patch1])
    plt.show()

    print(cm)

def main():
    df = get_data()
    features = ["voiceprint_mean", "speaker_num"]
    identify_speaker(df, features)

main()