import pandas as pd
from preprocessor import generate_processed_df
from vectorizer import generate_vector_features
from naive_bayes_trainer import run_naive_bayes_experiments
from train_FFNN import run_ffnn_experiments
import os
pd.options.mode.chained_assignment = None


def main():
    if os.path.isfile("./preprocessed_data.json"):
        df = pd.read_json("preprocessed_data.json")
    else:
        data = {"label": [], "sms": []}
        data_file = list(map(lambda x: x.split("\t"), open("SMS_Spam_Collection/SMSSpamCollection.txt").read()[:-1].split("\n")))

        for i in data_file:
            data["label"].append(i[0])
            data["sms"].append(i[1])

        df = pd.DataFrame.from_dict(data)
        generate_processed_df(df)


    # comment out if you want to generate vectorizations
    # generate_vector_features(df)

    run_naive_bayes_experiments("./vectorizations/")
    run_ffnn_experiments("./vectorizations/")

    #with open('./vectorizations/countvectorizer_ngrams_2_2/sms.pickle', 'rb') as handle:
    #    b = pickle.load(handle)


if __name__ == '__main__':
    main()






