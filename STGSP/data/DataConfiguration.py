class DataConfigurationBikeNYC():
    name = 'BikeNYC'
    dataset = 'bike_flow.npy'

    train_ratio = 0.9

    batch_size = 32

    len_closeness = 3
    len_period = 4
    len_trend = 4
    len_seq = len_closeness + len_period + len_trend

    T = 24 * 2
    T_closeness = 1
    T_period = T
    T_trend = T * 7

    days_test = 20
    len_test = T * days_test

    dim_flow = 2
    dim_h = 10
    dim_w = 20

class DataConfigurationTaxiNYC():
    name = 'TaxiNYC'
    dataset = 'taxi_flow.npy'

    train_ratio = 0.9

    batch_size = 32

    len_closeness = 3
    len_period = 4
    len_trend = 4
    len_seq = len_closeness + len_period + len_trend

    T = 24 * 2
    T_closeness = 1
    T_period = T
    T_trend = T * 7

    days_test = 20
    len_test = T * days_test

    dim_flow = 2
    dim_h = 10
    dim_w = 20

class DataConfigurationTaxiBJ():
    name = 'TaxiBJ'
    dataset = 'BJ13_M32x32_T30_InOut.h5'

    train_ratio = 0.9

    batch_size = 32

    len_closeness = 3
    len_period = 4
    len_trend = 4
    len_seq = len_closeness + len_period + len_trend

    T = 24 * 2
    T_closeness = 1
    T_period = T
    T_trend = T * 7

    days_test = 20
    len_test = T * days_test

    dim_flow = 2
    dim_h = 32
    dim_w = 32