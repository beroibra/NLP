from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus



def main():
    data_folder = "../5fold_stratified_folds/"
    model_dir = "./model_bert_base_cased_5_fold/"
    transformer_name = "bert-base-cased"

    for i in range(5):
        print(i+1)
        print(data_folder + str(i+1) + "/")
        print(model_dir + str(i+1) + "/")

        # load corpus containing training, test and dev data and if CSV has a header, you can skip it
        corpus: Corpus = CSVClassificationCorpus(data_folder + str(i+1) + "/", label_type="status",
                                                 column_name_map={0: "label_status", 1: "text"},
                                                 delimiter='\t', train_file="train.csv",
                                                 dev_file="dev.csv",
                                                 test_file="test.csv")

        weight_dict = {'__label__ham': 0.4, '__label__spam': 0.6}

        label_dict_csv = corpus.make_label_dictionary("status")

        document_embeddings = TransformerDocumentEmbeddings(transformer_name, fine_tune=True)

        classifier = TextClassifier(document_embeddings, label_type="status", label_dictionary=label_dict_csv, loss_weights=weight_dict)

        trainer = ModelTrainer(classifier, corpus)

        trainer.fine_tune(model_dir + str(i+1) + "/", learning_rate=3e-5, mini_batch_size=16, max_epochs=4,
                          train_with_dev=True, monitor_train=True, monitor_test=True)


if __name__ == '__main__':
    main()



"""
srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=64G -p batch   --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER,/ds:/ds:ro,`pwd`:`pwd`   --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.04-py3.sqsh   --container-workdir=`pwd`   --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" ./run_binary_doc_class.sh

"""