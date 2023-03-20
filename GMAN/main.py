import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn


from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=11,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=1,
                    help='prediction steps')
parser.add_argument('--in_dim', type=int, default=2,
                    help='num of flow type')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=1000,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')


parser.add_argument('--dataset', default='BikeNYC', help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--device', type=int, default = 1, help = 'gpu device')
parser.add_argument('--train_ratio', type=float, default=0.9, help='training set [default : 0.6]')
parser.add_argument('--T', type=int, default=48)
parser.add_argument('--days_test', type=int, default=20)
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=4)
parser.add_argument('--len_trend', type=int, default=4)

args = parser.parse_args()

if args.dataset == 'BikeNYC':
    args.traffic_file = './data/BikeNYC/bike_flow.npy'
    args.SE_file = './data/BikeNYC/SE(BikeNYC).txt'
    args.start_time = '20160701'
    args.model_file = './exps/BikeNYC/GMAN.pkl'
    args.log_file = './exps/BikeNYC/log'
elif args.dataset == 'TaxiNYC':
    args.traffic_file = './data/TaxiNYC/taxi_flow.npy'
    args.SE_file = './data/TaxiNYC/SE(TaxiNYC).txt'
    args.start_time = '20150101'
    args.model_file = './exps/TaxiNYC/GMAN.pkl'
    args.log_file = './exps/TaxiNYC/log'
elif args.dataset == 'TaxiBJ':
    args.traffic_file = './data/TaxiBJ/BJ13_M32x32_T30_InOut.h5'
    args.SE_file = './data/TaxiBJ/SE(TaxiBJ).txt'
    args.start_time = '20130101'
    args.model_file = './exps/TaxiBJ/GMAN.pkl'
    args.log_file = './exps/TaxiBJ/log'
else:
    raise ValueError('dataset does not exist')

seed = 777

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std

# build model
log_string(log, 'compiling model...')

model = GMAN(device, SE, args, bn_decay=0.1).to(device)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(device, model, args, log, loss_criterion, optimizer, scheduler)
    trainPred, valPred, testPred, test_score = test(device, args, log)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()


