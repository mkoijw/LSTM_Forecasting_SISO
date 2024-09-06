from data_process import nn_seq_us
from args import args_parser
from model import train ,test
import os

args = args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
if args.inout_mode == 'SISO':
    file_name = 'sig_data.csv'
    LSTM_PATH = path + '/LSTM_Learning/model/univariate_single_step.pkl'
else:
    file_name = 'mult_data.csv'
    LSTM_PATH = path + '/LSTM_Learning/model/multvariate_single_step.pkl'


dtr, val, dte, m, n = nn_seq_us(args,args.batch_size,file_name)
train(args, dtr, val, LSTM_PATH)
test(args, dte, LSTM_PATH, m, n)