import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import numpy as np
from itertools import chain
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq[0], input_seq[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred

# def get_val_loss(model, data):
#     model.eval()
#     out = model(data)
#     loss_function = torch.nn.CrossEntropyLoss().to(device)
#     loss = loss_function(out[data.val_mask], data.y[data.val_mask])
#     _, pred = out.max(dim=1)
#     model.train()
#     return loss.item()


def get_val_loss(model, val_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    with torch.no_grad():  # 在评估模式下不计算梯度
        for inputs, targets in val_loader:  # 遍历 DataLoader 中的每个批次
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)  # 模型预测

            # 计算损失
            loss_function = torch.nn.CrossEntropyLoss().to(device)
            loss = loss_function(out, targets)
            total_loss += loss.item()

            # 计算准确率
            _, pred = out.max(dim=1)
            correct += (pred == targets).sum().item()

    avg_loss = total_loss / len(val_loader)  # 平均损失
    accuracy = correct / len(val_loader.dataset)  # 准确率
    model.train()  # 恢复训练模式

    return avg_loss, accuracy


def train(args, dtr, val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size

    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss,accuracy = get_val_loss(model, val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))

def test(args, dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size

    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()