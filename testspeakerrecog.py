import os
import numpy as np
import pandas as pd

from audio_data import get_data
import scipy.io.wavfile as wavf
from python_speech_features import mfcc, logfbank
from scipy.signal import stft
import pickle


import matplotlib.pyplot as plt

def get_person_dir(person, data_dir):
    return os.path.join(data_dir,person)

def plot_result(data,predict,seddf, shawndf, unknowndf, yeshadf):
    plt.title('Classification For Input Data')
    plt.scatter(seddf['mfcc_mean'], seddf['fbankmean'], color='blue', label='Sedrick')
    plt.scatter(yeshadf['mfcc_mean'], yeshadf['fbankmean'], color='red', label='Yesha')
    plt.scatter(shawndf['mfcc_mean'], shawndf['fbankmean'], color='green', label='Shawn')
    plt.scatter(unknowndf['mfcc_mean'], unknowndf['fbankmean'], color='orange', label='Unknown')
    plt.scatter(data['mfcc_mean'], data['fbankmean'], color='black', label='Input Data Point')
    plt.text(data['mfcc_mean'] + 0.5, data['fbankmean'], predict, bbox=dict(facecolor='white', alpha=0.5))
    plt.ylabel('Filter Bank')
    plt.xlabel('MFCC')
    plt.legend()
    plt.show()


def plot_result1(seddf, shawndf, unknowndf, yeshadf):
    plt.title('Classification For Input Data')
    plt.scatter(seddf['mfcc_mean'], seddf['fbankmean'], color='blue', label='Sedrick')
    plt.scatter(yeshadf['mfcc_mean'], yeshadf['fbankmean'], color='red', label='Yesha')
    plt.scatter(shawndf['mfcc_mean'], shawndf['fbankmean'], color='green', label='Shawn')
    plt.scatter(unknowndf['mfcc_mean'], unknowndf['fbankmean'], color='orange', label='Unknown')
    plt.ylabel('Filter Bank')
    plt.xlabel('MFCC')
    plt.legend()
    plt.show()



from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, confusion_matrix, precision_recall_curve



svc = SVC(kernel="linear")
tree = DecisionTreeClassifier()
neighbors = KNeighborsClassifier(n_neighbors=8)
nb = GaussianNB()
rf = RandomForestClassifier()

# def clean_df(df, features):
#     for feature in features:
#         df[feature] = df[feature].apply(lambda x: open_pickle(x))
#     return df
#
# def open_pickle(file):
#     eq = file.index("=") + 2
#     file = file[eq:-2]
#     obj = pd.read_pickle(file)
#     obj = obj.flatten()
#     return obj
#
# def join_features(df, features, columns):
#     for column in columns:
#         new_column = "final_" + column
#         for feature in features:
#             df[new_column] = df.apply(lambda x: np.append(x[column], x[feature]), axis=1)
#     return df
#
# def get_result(result, speaker, score):
#     if result == speaker:
#         score += 1
#     speakers = ["sedrick", "shawn", "yesha"]
#
#     return score, speakers[result]

def train_test_models(traindf,testdf, models, features, feature_names):
    target = ["speaker_num"]
    X = traindf[features].values
    y = traindf[target].values
    X_unseen = testdf[features].values
    y_unseen = testdf[target].values
    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    #
    # print("X test shape: ", X_test.shape)
    # print("y test shape: ", y_test.shape)

    scores = []

    for model in models:
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        y_unseen_pred = model.predict(X_unseen)

        scores.append([
            str(model), accuracy_score(y_test, y_pred), accuracy_score(y_unseen, y_unseen_pred)
        ])

        # scores.append([
        #     str(model), model.score(X_train, y_train), model.score(X_test, y_test),
        #     accuracy_score(y_test, y_pred), accuracy_score(y_unseen, y_unseen_pred)
        # ])

    scores = pd.DataFrame(scores, columns=["model", "test", "unseen"])
    print("Here are the scores for the models when testing training on the following featues:", feature_names)
    print(scores)
    print()
    print()

    return models


    # for i in range(len(testdf)):
    #     df = testdf.loc[[i]]
    #     speaker = df.speaker.values[0]
    #     print(df)
    #     print()
    #     print("Testing speaker:", speaker)
    #     # plot_result(df, seddf, shawndf, yeshadf)
    #
    #     result = svc.predict(df[features])[0]
    #     svcscore, result = get_result(result, speaker, svcscore)
    #     print("svc result:", result)
    #
    #     result = tree.predict(df[features])[0]
    #     treescore, result = get_result(result, speaker, treescore)
    #     print("tree result:", result)
    #
    #     result = neighbors.predict(df[features])[0]
    #     neighborsscore, result = get_result(result, speaker, neighborsscore)
    #     print("neighbors result:", result)
    #
    #     result = nb.predict(df[features])[0]
    #     nbscore, result = get_result(result, speaker, nbscore)
    #     print("nb result:", result)
    #
    #     result = rf.predict(df[features])[0]
    #     nbscore, result = get_result(result, speaker, rfscore)
    #     print("rf result:", result)
    #     # probability = clf.predict_proba(randdf[features])
    #     print()
    #     print()
    #
    # print("svc score: ", svcscore)
    # print("tree score: ", treescore)
    # print("neighbors score: ", neighborsscore)
    # print("nb score: ", nbscore)
    # print("rf score: ", rfscore)




df = get_data()
target = ["speaker_num"]

df = df.loc[df["authentic"] == 1]




#
# unknown = df[df["filename"].str.contains("unknown")].head()
#
# print(unknown["person"])


seddf = df.loc[df["speaker_num"] == 0].sample(frac=1)
trainsed = seddf.head(len(seddf) - 10)
testsed = seddf.tail(10)
# print(len(seddf))
# # print(len(seddf))
# # print(seddf.index)
# # testsed = seddf.sample(frac=0.45)
# # print(len(testsed))
# # testsed = seddf.loc[185:194]
# # seddf = seddf.loc[0:184]
# # print(len(seddf))
# #
shawndf = df.loc[df["speaker_num"] == 2].sample(frac=1)
trainshawn = shawndf.head(len(shawndf) - 10)
testshawn = shawndf.tail(10)
# print(len(shawndf))
# # print(len(shawndf))
# # print(shawndf.index)
# # testshawn = shawndf.sample(frac=0.45)
# # print(len(testshawn))
# # testshawn = shawndf.loc[378:387]
# # shawndf = shawndf.loc[195:377]
# # print(len(shawndf))
# #
# #
# #
unknowndf = df.loc[(df["speaker_num"] == 1) & (df["fbankmean"] > 1)].sample(frac=1)
trainunknown = unknowndf.head(len(unknowndf) - 10)
testunknown = unknowndf.tail(10)
print(len(unknowndf))
# print(len(unknowndf))
# print(unknowndf.index)


yeshadf = df.loc[df["speaker_num"] == 3].sample(frac=1)
trainyesha = yeshadf.head(len(yeshadf) - 10)
testyesha = yeshadf.tail(10)
# print(len(yeshadf))
# print(len(yeshadf))
# print(yeshadf.index)
# testyesha = yeshadf.sample(frac=0.45)
# print(len(testyesha))
# testyesha = yeshadf.loc[567:576]
# yeshadf = yeshadf.loc[388:566]
# print(len(yeshadf))


traindf = pd.concat([trainsed, trainshawn, trainunknown, trainyesha])
testdf = pd.concat([testsed, testshawn, testunknown, testyesha]).reset_index()


models = [svc, tree, neighbors, nb, rf]

features1 = ["fbankmean", "mfcc_mean"]
features1_names = ["fbankmean", "mfcc_mean"]

features2 = [str(i) for i in range(1,102)]
features2_names = ["STFT"]

features3 = features2 + features1
features3_names = ["SFTT", "fbankmean", "mfcc_mean"]

features4 = features2 + [features1[0]]
features4_names = ["SFTT", "fbankmeam"]

features5 = features2 + [features1[1]]
features5_names = ["SFTT", "mfcc_mean"]

f_list = [features1, features2, features3, features4, features5]
fname_list = [features1_names, features2_names, features3_names, features4_names, features5_names]

# plot_result(unknowndf, seddf, shawndf, yeshadf)

for i in range(5):
    f = f_list[i]
    fn = fname_list[i]
    train_test_models(traindf, testdf, models,f, fn)







# fft = [str(i) for i in range(1,102)]
# features = ["fbankmean", "mfcc_mean"]
# features.extend(fft)
#
#
# plot_result(unknowndf, seddf, shawndf, yeshadf)
#
# svc.fit(X_train, y_train)
# y_predsvc = svc.predict(X_test)
# score = svc.score(X_test, y_test)
# print("svc score: ", score)
#
# tree.fit(X_train, y_train)
# y_predsvc = tree.predict(X_test)
# score = tree.score(X_test, y_test)
# print("tree score: ", score)
#
# neighbors.fit(X_train, y_train)
# y_predsvc = neighbors.predict(X_test)
# score = neighbors.score(X_test, y_test)
# print("neighbors score: ", score)
#
# nb.fit(X_train, y_train)
# y_prednb = nb.predict(X_test)
# score = nb.score(X_test, y_test)
# print("nb score: ", score)
#
#
# rf.fit(X_train, y_train)
# y_prednb = rf.predict(X_test)
# score = rf.score(X_test, y_test)
# print("rf score: ", score)
#
#
# print()
# print()
#
# svcscore = 0
# neighborsscore = 0
# treescore = 0
# nbscore= 0
# rfscore = 0




#

# temp_dir = "/Users/sedrickcashawjr/Documents/School/ECE4983/AudioFraud/AudioFraud/data/unknown"
# for index, filename in enumerate(os.listdir(temp_dir)):
#     dst = "unknownfile"+ str(index+1) + ".wav"
#     src = temp_dir + "/" + filename
#     dst = temp_dir + "/" + dst
#     os.rename(src, dst)

    # for i in range(30):
    #     df = testdf.loc[[i]]
    #     speaker = df.speaker.values[0]
    #     print(df)
    #     print()
    #     print("Testing speaker:", speaker)
    #     # plot_result(df, seddf, shawndf, yeshadf)
    #
    #     result = svc.predict(df[features])[0]
    #     svcscore, result = get_result(result, speaker, svcscore)
    #     print("svc result:", result)
    #
    #     result = tree.predict(df[features])[0]
    #     treescore, result = get_result(result, speaker, treescore)
    #     print("tree result:", result)
    #
    #     result = neighbors.predict(df[features])[0]
    #     neighborsscore, result = get_result(result, speaker, neighborsscore)
    #     print("neighbors result:", result)
    #
    #     result = nb.predict(df[features])[0]
    #     nbscore, result = get_result(result, speaker, nbscore)
    #     print("nb result:", result)
    #
    #     result = rf.predict(df[features])[0]
    #     nbscore, result = get_result(result, speaker, rfscore)
    #     print("rf result:", result)
    #     # probability = clf.predict_proba(randdf[features])
    #     print()
    #     print()
    #
    # print("svc score: ", svcscore)
    # print("tree score: ", treescore)
    # print("neighbors score: ", neighborsscore)
    # print("nb score: ", nbscore)
    # print("rf score: ", rfscore)

def make_pickle(filename, obj):
    pickle_out = open(filename, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def get_speaker(speaker):
    speakers = {0: "Sedrick",
                1: "Unknown",
                2: "Shawn",
                3:"Yesha"}

    return speakers[speaker]


def identify_speaker(model,input):
    # model = pd.read_pickle("data/classifiers/voice_model")
    # features = ["fbankmean", "mfcc_mean"]
    features = [i for i in range(1,102)] + ["fbankmean", "mfcc_mean"]
    input = input[features]
    result = model.predict(input)[0]
    speaker = get_speaker(result)
    return speaker

def extract_features(file):
    row = {}
    audio_file = open(file, "rb")

    rate, stereo = wavf.read(audio_file)
    if len(stereo) == 0:
        return
    if not isinstance(stereo[0], np.ndarray):
        mono = stereo
    else:
        mono = stereo[:, 0]

    row["rate"] = [rate]

    f, t, Zxx = stft(mono, rate, nperseg=200)
    temp = pd.DataFrame(Zxx.T).abs().mean().values
    count = 1
    for val in temp:
        row[count] = [val]
        count += 1

    fbank_mean = np.mean(logfbank(stereo, rate, nfft=1200))
    row["fbankmean"] = [fbank_mean]

    mfcc_mean = np.mean(mfcc(stereo, rate, nfft=1200))
    row["mfcc_mean"] = [mfcc_mean]

    return pd.DataFrame(row)

# test = testdf.sample(1)

# plot_result1(trainsed, trainshawn, trainunknown, trainyesha)

models = train_test_models(traindf, testdf, models, features3, features3_names)
# plot_result(data, result, trainsed, trainshawn, trainunknown, trainyesha)


count = 1
for model in models:
    new_name = "data/classifiers/speaker_recognition_model" + str(count)
    make_pickle(new_name, model)
    count += 1


# path = "/Users/sedrickcashawjr/Documents/School/ECE4983/AudioFraud/AudioFraud/unseen"
# for file in os.listdir(path):
#     file = path + "/" + file
#     print(file)
#     input = extract_features(file)
#     model = models[1]
#     result = identify_speaker(model, input)
#     plot_result(input, result, trainsed, trainshawn, trainunknown, trainyesha)

    # results = [None] * len(models)
    # for model in models:
    #     result = identify_speaker(model, input)
    #     model_index = models.index(model)
    #     results[model_index] = result
    # print(results)
    # print()
    # print()
















