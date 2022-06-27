from torch.utils.data import DataLoader, Dataset
import torch


class SmsSpamDataset(Dataset):
    def __init__(self, sms, labels):
        self.data = torch.from_numpy(sms)
        self.labels = labels
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        curr_sms = self.data[idx]

        return curr_sms, self.labels[idx]


def generate_dataloaders(train_dataset, test_dataset):
    train_params = {'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 6}

    test_params = {'batch_size': 256,
                   'shuffle': True,
                   'num_workers': 6}

    return DataLoader(train_dataset, **train_params), DataLoader(test_dataset, **test_params)
