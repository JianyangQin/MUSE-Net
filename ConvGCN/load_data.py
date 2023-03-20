from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import csv
import h5py

global_start_time = time.time()
def Get_All_Data(TG,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
	# deal with inflow data 处理进站数据
	metro_enter = []
	with open('data/inflowdata/in_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line=[int(x) for x in line]
			metro_enter.append(line)

	def get_train_data_enter(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data=np.array(data)
		data2=np.zeros((data.shape[0],data.shape[1]))
		a=np.max(data)
		b=np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i,j]=round((data[i,j]-b)/(a-b),5)
		# 不包括第一周和最后一周的数据
		# not include the first week and the last week among the five weeks
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		Y_train = []
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_train_1[index-TG_in_one_week].append(temp)
			Y_train.append(data2[:,index + time_lag-1])
		X_train_1,Y_train = np.array(X_train_1), np.array(Y_train)
		print("X_train_1.shape,Y_train.shape")
		print(X_train_1.shape,Y_train.shape)

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		Y_test = []
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
			Y_test.append(data2[:,index + time_lag-1])
		X_test_1,Y_test = np.array(X_test_1), np.array(Y_test)
		print("X_test_1.shape,Y_test.shape")
		print(X_test_1.shape,Y_test.shape)

		Y_test_original = []
		for index in range(len(data[0]) - TG_in_one_day*forecast_day_number,len(data[0])-time_lag+1):
			Y_test_original.append(data[:,index + time_lag-1])
		Y_test_original = np.array(Y_test_original)
		print("Y_test_original.shape")
		print(Y_test_original.shape)

		return X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b

	#获取训练集和测试集，Y_test_original为没有scale之前的原始测试集，评估精度用，a,b分别为最大值和最小值
	#Get the training dataset and the test dataset, Y_test_original is the original test data before scaling, which can be used for evaluation.
	#a and b as the maximum and minimum values, respectively.
	X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b=get_train_data_enter(metro_enter,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week)
	print(a,b)

	#deal with outflow data. Similar with the inflow data while not including the testing data for outflow
	#处理出站数据
	metro_exit = []
	with open('data/outflowdata/out_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line=[int(x) for x in line]
			metro_exit.append(line)

	def get_train_data_exit(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data=np.array(data)
		data2=np.zeros((data.shape[0],data.shape[1]))
		a=np.max(data)
		b=np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i,j]=round((data[i,j]-b)/(a-b),5)
		#不包括第一周和最后一周
		## not include the first week and the last week among the five weeks
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()#上周同一个时间段的数据
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])#前一天同一个时间段的数据
				temp.extend(data2[i,index: index + time_lag-1])#当天前几个时间段的数据
				X_train_1[index-TG_in_one_week].append(temp)
		X_train_1= np.array(X_train_1)
		print(X_train_1.shape)#其形状应该是(sample number, 276, 5, channel=3),3代表着上一周，前一天，当天，相当于275*5*3的图片

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			#此处注意test的下标要从0开始，而data2_all的下标要从
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
		X_test_1= np.array(X_test_1)
		print(X_test_1.shape)
		return X_train_1,X_test_1

	X_train_2,X_test_2=get_train_data_exit(metro_exit,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week)


	return X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2

class MM:
    def __init__(self, MM_max, MM_min):
        self.max = MM_max
        self.min = MM_min

def load_npy_data(dataset, train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness=1, T_period=24, T_trend=24 * 7):
    all_data = np.load(dataset)
    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mm = MM(np.max(all_data), np.min(all_data))
    print('max=', mm.max, ' min=', mm.min)

    all_data = (2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=', np.mean(all_data), ' variance=', np.std(all_data))
    all_data = all_data.reshape((all_data.shape[0], all_data.shape[1], -1))

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    elif len_period > 0:
        number_of_skip_hours = T_period * len_period
    elif len_closeness > 0:
        number_of_skip_hours = T_closeness * len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - len_test) * train_ratio)
    len_val = len(Y) - len_train - len_test

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train+len_val]
    X_period_val = X_period[len_train:len_train+len_val]
    X_trend_val = X_trend[len_train:len_train+len_val]

    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_val = [X_closeness_val, X_period_val, X_trend_val]
    X_test = [X_closeness_test, X_period_test, X_trend_test]

    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_val = np.concatenate((X_val[0], X_val[1], X_val[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

    X_train = np.transpose(X_train.reshape((X_train.shape[0], -1, 2, X_train.shape[-1])), (0, 3, 1, 2))
    X_val = np.transpose(X_val.reshape((X_val.shape[0], -1, 2, X_val.shape[-1])), (0, 3, 1, 2))
    X_test = np.transpose(X_test.reshape((X_test.shape[0], -1, 2, X_test.shape[-1])), (0, 3, 1, 2))

    Y_train = np.transpose(Y[:len_train], (0, 2, 1))
    Y_val = np.transpose(Y[len_train:len_train+len_val], (0, 2, 1))
    Y_test = np.transpose(Y[-len_test:], (0, 2, 1))


    print('len_train=' + str(len_train))
    print('len_val=' + str(len_val))
    print('len_test =' + str(len_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mm.max - mm.min, mm.max, mm.min

def remove_incomplete_days(data, timestamps, T=48):
    """
    remove a certain day which has not 48 timestamps
    :param data:
    :param timestamps:
    :param T:
    :return:
    """

    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def load_taxibj(dataset, train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness=1, T_period=24, T_trend=24 * 7, T=48):
    f = h5py.File(dataset, 'r')
    all_data = f['data'].value
    timestamps = f['date'].value
    f.close()
    all_data, _ = remove_incomplete_days(all_data, timestamps, T)

    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mm = MM(np.max(all_data), np.min(all_data))
    print('max=', mm.max, ' min=', mm.min)

    all_data=(2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=', np.mean(all_data), ' variance=', np.std(all_data))
    all_data = all_data.reshape((all_data.shape[0], all_data.shape[1], -1))

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    elif len_period > 0:
        number_of_skip_hours = T_period * len_period
    elif len_closeness > 0:
        number_of_skip_hours = T_closeness * len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - len_test) * train_ratio)
    len_val = len(Y) - len_train - len_test

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train + len_val]
    X_period_val = X_period[len_train:len_train + len_val]
    X_trend_val = X_trend[len_train:len_train + len_val]

    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_val = [X_closeness_val, X_period_val, X_trend_val]
    X_test = [X_closeness_test, X_period_test, X_trend_test]

    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_val = np.concatenate((X_val[0], X_val[1], X_val[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

    X_train = np.transpose(X_train.reshape((X_train.shape[0], -1, 2, X_train.shape[-1])), (0, 3, 1, 2))
    X_val = np.transpose(X_val.reshape((X_val.shape[0], -1, 2, X_val.shape[-1])), (0, 3, 1, 2))
    X_test = np.transpose(X_test.reshape((X_test.shape[0], -1, 2, X_test.shape[-1])), (0, 3, 1, 2))

    Y_train = np.transpose(Y[:len_train], (0, 2, 1))
    Y_val = np.transpose(Y[len_train:len_train + len_val], (0, 2, 1))
    Y_test = np.transpose(Y[-len_test:], (0, 2, 1))

    print('len_train=' + str(len_train))
    print('len_val=' + str(len_val))
    print('len_test =' + str(len_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mm.max - mm.min, mm.max, mm.min

def generate_adjacency_matrix(col, row):
    adj_mx = np.zeros((col*row, col*row))
    for i in range(col * row):
        for j in range(col * row):
            if j == i:
                adj_mx[i][j] = 1.0
            elif (j == i + 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif j == i - col:
                adj_mx[i][j] = 1.0
            elif j == i + col:
                adj_mx[i][j] = 1.0
            elif (j == i - col - 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - col + 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col - 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col + 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            else:
                adj_mx[i][j] = 0.0
    return adj_mx