from data_process import nn_seq_us
from args import args_parser
from model import train ,test
import os

file_name = 'sig_data.csv'
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/LSTM_c/model/univariate_single_step.pkl'

dtr, val, dte, m, n = nn_seq_us(10,file_name)
args = args_parser()
train(args, dtr, val, LSTM_PATH)
test(args, dte, LSTM_PATH, m, n)