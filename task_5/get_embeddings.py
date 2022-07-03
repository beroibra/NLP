import itertools

from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np
from scipy import spatial
from sklearn.metrics import pairwise_distances
import pandas as pd


def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def get_embeddings(df, transformer, fine_tuned=False):

    if fine_tuned:
        embedder = TextClassifier.load(transformer).document_embeddings
    else:
        embedder = TransformerDocumentEmbeddings(transformer)

    sms = df["sms"].to_list()
    labels = df["label"].to_list()
    labels_num = [0 if i == "ham" else 1 for i in labels]

    res = {i: [] for i in list(set(labels_num))}
    idx = 0
    for curr_sms_label in zip(labels_num, sms):
        print(idx, "/", len(labels_num))
        sent = Sentence(curr_sms_label[1])
        embedder.embed(sent)
        res[curr_sms_label[0]].append(sent.embedding.detach().numpy())
        idx += 1

    min_extracted_sents = min([len(res[i]) for i in list(res.keys())])

    for i in res:
        res[i] = np.array(res[i])

    """
    
    for i in range(len(all_cor)):
        sent = all_cor[i]
        embedder.embed(sent)
        res[int(all_cor[i].annotation_layers['sentiment'][0].value.split("__")[-1])].append(sent)
        print()
        #res_labels[int(all_cor[i].annotation_layers['sentiment'][0].value.split("__")[-1])] += 1

        #temp_check = [True if res_labels[j] == min_class_occ else False for j in res_labels]
        #if len(list(set(temp_check))) == 1:
        #    if temp_check[0]:
        #        break

    min_extracted_sents = min([len(res[i]) for i in list(res.keys())])

    for i in res:
        res[i] = np.array([sent.embedding.detach().numpy() for sent in res[i]])[:min_extracted_sents]
    """

    return res


def get_distance_matrix(class_embeddings):
    for res_comb in sorted([(i, i) for i in range(len(class_embeddings))] + list(
            itertools.combinations(range(len(class_embeddings)), 2)), key=lambda x: x[0] + x[1]):
        class_o = class_embeddings[res_comb[0]]
        class_i = class_embeddings[res_comb[1]]

        #pdistances_cosine = pairwise_distances(class_o, class_i, metric='cosine', n_jobs=-1)
        pdistances_cosine = pairwise_distances(class_o, class_i, metric=cosine_similarity, n_jobs=-1)
        mean_dist_cosine = pdistances_cosine.mean()
        print("Cosine distance between classes {} and {}: {}".format(res_comb[0], res_comb[1], mean_dist_cosine))


df = pd.read_json("../preprocessed_data.json")


embeddings_bef_ft = get_embeddings(df, "bert-base-cased", fine_tuned=False)

embeddings_aft_ft = get_embeddings(df, "./ft_bert_base_cased_model.pt", fine_tuned=True)

print("Mean distance Matrix before Fine-tuning:\n")
get_distance_matrix(embeddings_bef_ft)
print("-----------------------------\n\n")

print("Mean distance Matrix after Fine-tuning:\n")
get_distance_matrix(embeddings_aft_ft)
print("-----------------------------\n\n")





