##Import classifer models
from sklearn.linear_model import LogisticRegression

#Feature Manuiplaton
from sklearn.model_selection import train_test_split

#Import metrics
from sklearn.metrics import confusion_matrix


def extract_features():
    None

def identify_speaker(df, X_features, y_features):
    log_regr = LogisticRegression(random_state=0)
    X = df.iloc[:, X_features:].values
    y = df.iloc[:, y_features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    log_regr.fit(X_train, y_train)
    y_pred = log_regr.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
