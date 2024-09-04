import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(file_name):
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    return df

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def nn_seq_us(b,file_name):
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        # load = data[data.columns[1]]
        load = np.array(data[data.columns[1]])
        load = np.array(load.tolist())
        data = data.values.tolist()
        load = (load - n) / (m - n)
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        return seq

    dtr = process(train, b)
    val = process(val, b)
    dte = process(test,b)

    return dtr, val, dte, m, n