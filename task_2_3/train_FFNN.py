import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import dataloader
from FFNN import FFNN
import torch
import os


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)#.squeeze().numpy()
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, device, l1_norm):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x.float())

        loss = criterion(y_pred, y)

        #if l1_norm:
        loss += 0.000001 * sum(p.abs().sum() for p in model.parameters())

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        #prevent_negative_output = True
        #if prevent_negative_output:
        #    with torch.no_grad():
        #        model.layers[-1].weight.data = model.layers[-1].weight.data.clamp(0)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x.float())

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_predictions(model, iterator, device, path, curr_file_name):
    model.eval()

    labels = []
    probs = []

    with torch.no_grad():
        for step, (x, y) in enumerate(iterator):
            x = x.to(device)

            y_pred = model(x.float())

            # y_prob = F.softmax(y_pred, dim=-1)
            # top_pred = y_prob.argmax(1, keepdim=True)

            labels.append(y.cpu())
            probs.append(y_pred.cpu())

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    curr_res = open(path+curr_file_name+"_res.txt", "w+")
    print(classification_report(labels.numpy(), probs.argmax(1, keepdim=True).squeeze().numpy(), target_names=["ham", "spam"]))
    curr_res.write(classification_report(labels.numpy(), probs.argmax(1, keepdim=True).squeeze().numpy(), target_names=["ham", "spam"]))
    ConfusionMatrixDisplay.from_predictions(labels.numpy(), probs.argmax(1, keepdim=True).squeeze().numpy(), labels=[0, 1], display_labels=["ham", "spam"])
    plt.savefig(path+curr_file_name+"_cf_matrix.png")
    plt.clf()
    #plt.show()
    return labels, probs


def train_loop(model, train_loader, optimizer, criterion, device, l1_norm, path, file_name):
    train_losses = []
    train_accs = []

    best_train_loss = float('inf')
    for epoch in range(15):
        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, l1_norm)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if os.path.isfile("./best_model.pt"):
                os.remove("./best_model.pt")

            torch.save(model.state_dict(), "./best_model.pt")

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        #print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    plt.plot(train_losses, label="Training loss")
    plt.legend()
    plt.savefig(path+file_name+"_train_loss.png")
    plt.clf()
    #plt.show()

    plt.plot(train_accs, label="Training accuracy")
    plt.legend()
    plt.savefig(path+file_name+"_train_acc.png")
    plt.clf()
    #plt.show()


def test_loop(model, test_iterator, criterion, device, path, curr_file_name):
    model.load_state_dict(torch.load('./best_model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    labels, probs = get_predictions(model, test_iterator, device, path, curr_file_name)
    pred_labels = torch.argmax(probs, 1)


def train_ffnn(data, exp_name, vec_setup, vec):
    path = "./Models/FFNN/"+exp_name+"/"+vec_setup+"/"
    curr_file_name = vec.split(".")[0]
    os.makedirs(path, exist_ok=True)

    with open(data, 'rb') as handle:
        curr_data = pickle.load(handle)

    y = np.array([0 if i == 'ham' else 1 for i in curr_data['label']])
    X_train, X_test, y_train, y_test = train_test_split(curr_data['vectors'], y, test_size=0.2, random_state=42,
                                                        shuffle=True, stratify=y)
    train_loader, test_loader = dataloader.generate_dataloaders(
        dataloader.SmsSpamDataset(X_train.astype(float), y_train),
        dataloader.SmsSpamDataset(X_test.astype(float), y_test))

    model = FFNN(X_train.shape[1], len(set(curr_data['label'])))
    optimizer = model.get_optimizer()
    criterion = model.get_loss_function()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = criterion.to(device)

    train_loop(model, train_loader, optimizer, criterion, device, False, path, curr_file_name)
    test_loop(model, test_loader, criterion, device, path, curr_file_name)

    print()


def run_ffnn_experiments(data_path):
    vec_setups = ["countvectorizer_ngrams_2_2", "tfidfvectorizer", "fasttext_vec", "glove_vec", "albert_vec"]
    #vec_setups = ["glove_vec"]
    vecs = ["sms.pickle", "preprocessing.pickle", "swr.pickle", "freq_rare_word_rm.pickle", "stemmed.pickle",
            "lemmatized.pickle"]
    #vecs = ["sms.pickle"]

    exp_name = "6_15_epochs_1e-4_lr_l1_l2"

    for vec_setup in vec_setups:
        print(vec_setup)
        print("-----------------")
        for vec in vecs:
            print(vec)
            train_ffnn(data_path + vec_setup + "/" + vec, exp_name, vec_setup, vec)
            print("-----------------")


if __name__ == '__main__':
    run_ffnn_experiments("./vectorizations/")
