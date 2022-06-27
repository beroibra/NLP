from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import sister
import os
from sentence_transformers import SentenceTransformer
import pickle

data_setups = ["sms", "preprocessing", "swr", "freq_rare_word_rm", "stemmed", "lemmatized"]

def count_vec(df):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    vectorizer_dir = "countvectorizer_ngrams_2_2"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)

    for setup in data_setups:
        curr_X = vectorizer.fit_transform(df[setup].to_list())
        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X.toarray()}
        with open('./vectorizations/' + vectorizer_dir + '/' + setup + '.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tfidf_vec(df):
    vectorizer = TfidfVectorizer()
    vectorizer_dir = "tfidfvectorizer"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)

    for setup in data_setups:
        curr_X = vectorizer.fit_transform(df[setup].to_list())
        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X.toarray()}
        with open('./vectorizations/'+vectorizer_dir+'/'+setup+'.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fasttext_vec(df):
    vectorizer = sister.MeanEmbedding(lang='en')
    vectorizer_dir = "fasttext_vec"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)

    for setup in data_setups:
        curr_X = np.array(list(map(lambda x: vectorizer(x), df[setup].to_list())))

        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X}
        with open('./vectorizations/' + vectorizer_dir + '/' + setup + '.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def bert_vec(df):
    vectorizer = sister.BertEmbedding(lang='en')
    vectorizer_dir = "albert_vec"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)

    for setup in data_setups:
        curr_X = np.array(list(map(lambda x: vectorizer(x), df[setup].to_list())))

        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X}
        with open('./vectorizations/' + vectorizer_dir + '/' + setup + '.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def glove_vec(df):
    vectorizer = SentenceTransformer('average_word_embeddings_glove.6B.300d')
    vectorizer_dir = "glove_vec"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)

    for setup in data_setups:
        print(setup)
        curr_X = np.array(list(map(lambda x: vectorizer.encode(x), df[setup].to_list())), dtype=object)

        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X}
        with open('./vectorizations/' + vectorizer_dir + '/' + setup + '.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sbert_mpnet_base_vec(df):
    vectorizer = SentenceTransformer('all-mpnet-base-v2')
    vectorizer_dir = "sbert_mpnet_base_vec"
    os.makedirs("./vectorizations/"+vectorizer_dir+"/", exist_ok=True)


    for setup in data_setups:
        print(setup)
        curr_X = np.array(list(map(lambda x: vectorizer.encode(x), df[setup].to_list())), dtype=object)

        res_dict = {"label": df["label"].to_list(), "sms": df[setup].to_list(), "vectors": curr_X}
        with open('./vectorizations/' + vectorizer_dir + '/' + setup + '.pickle', 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_vector_features(df):
    count_vec(df)
    tfidf_vec(df)
    #fasttext_vec(df)
    glove_vec(df)
    bert_vec(df)
    sbert_mpnet_base_vec(df)




