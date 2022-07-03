from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# ham --> 0
# spam --> 1

def train_naive_bayes(data, vec_setup, vec):

    with open(data, 'rb') as handle:
        curr_data = pickle.load(handle)

    y = np.array([0 if i == 'ham' else 1 for i in curr_data['label']])
    X_train, X_test, y_train, y_test = train_test_split(curr_data['vectors'], y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    os.makedirs("./Models/naive_bayes_trainer/"+vec_setup+"/", exist_ok=True)
    curr_res = open("./Models/naive_bayes_trainer/"+vec_setup+"/"+vec.split(".")[0]+"_res.txt", "w+")

    curr_res.write(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1], display_labels=["ham", "spam"])
    plt.savefig("./Models/naive_bayes_trainer/"+vec_setup+"/"+vec.split(".")[0]+"_cf_matrix.png")


def run_naive_bayes_experiments(data_path):
    vec_setups = ["countvectorizer_ngrams_2_2", "tfidfvectorizer", "fasttext_vec", "glove_vec", "albert_vec"]
    #vec_setups = ["glove_vec"]
    vecs = ["sms.pickle", "preprocessing.pickle", "swr.pickle", "freq_rare_word_rm.pickle", "stemmed.pickle", "lemmatized.pickle"]
    #vecs = ["sms.pickle"]

    for vec_setup in vec_setups:
        print(vec_setup)
        print("-----------------")
        for vec in vecs:
            print(vec)
            train_naive_bayes(data_path+vec_setup+"/"+vec, vec_setup, vec)
            print("-----------------")


if __name__ == '__main__':
    run_naive_bayes_experiments("./vectorizations/")