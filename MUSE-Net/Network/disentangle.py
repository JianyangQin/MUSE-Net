import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Activation,Dropout,BatchNormalization, Reshape, Subtract
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.optimizers import Adam
from keras.models import Model
from Network.base import *
from Network.loss import *
import Network.metrics as metrics

def split(x, start, end):
    return x[:, start:end]

def get_z(x):
    mu = x[0]
    log_var = x[1]
    std = tf.exp(log_var)
    batch = tf.shape(log_var)[0]
    dim = tf.shape(log_var)[1]
    eps = tf.random_normal([batch, dim])
    return mu + eps * std

def reparameterize(in_channel, batch_size):
    mu = Input(batch_shape=(batch_size, in_channel))
    log_var = Input(batch_shape=(batch_size, in_channel))
    z = Lambda(get_z)([mu, log_var])
    reparam = Model(inputs=[mu, log_var], outputs=z)
    return reparam

# Exclusive encoder for closeness, period and trend time sub-series
def Exclusive_Encoder(H=10, W=20, in_channel=6, out_channel=64, mu_dim=32, batch_size=8):
    cpt_input = Input(batch_shape=(batch_size, in_channel, H, W))

    cpt_conv = Conv2D(filters=out_channel, kernel_size=(1, 1), padding="same")(cpt_input)
    cpt_out = Flatten()(cpt_conv)
    cpt_out = Dense(units=mu_dim * 2)(cpt_out)

    mu = Lambda(split, arguments={'start': 0, 'end': mu_dim})(cpt_out)                      # First mu_dim bits of dense output can be regarded as mean (mu)
    log_var = Lambda(split, arguments={'start': mu_dim, 'end': mu_dim * 2})(cpt_out)        # Last mu_dim bits of dense output can be regarded as standard deviation (log_var)
    z = reparameterize(mu_dim, batch_size)([mu, log_var])                                   # Sample mu and log_var to generate the distribution of representation (z)

    es_encoder = Model(inputs=cpt_input, outputs=[mu, log_var, z, cpt_conv])
    return es_encoder

# 共享特征编码器
def Interactive_Feature_extractor(H=10, W=20, in_channel=6, out_channel=64, batch_size=8):
    cpt_input = Input(batch_shape=(batch_size, in_channel, H, W))

    cpt_out = Conv2D(filters=out_channel, kernel_size=(1, 1), padding="same")(cpt_input)

    feature_extractor = Model(inputs=cpt_input, outputs=cpt_out)
    return feature_extractor

# 单一时间分量变分编码器
def Simplex_Variational_Encoder(H=10, W=20,in_channel=64, out_channel=64, mu_dim=32, batch_size=8):
    cpt_input = Input(batch_shape=(batch_size, in_channel, H, W))

    cpt_out = Conv2D(filters=out_channel, kernel_size=(1, 1), padding="same")(cpt_input)
    cpt_dense = Flatten()(cpt_out)
    cpt_dense = Dense(units=mu_dim*2)(cpt_dense)

    mu = Lambda(split, arguments={'start': 0, 'end': mu_dim})(cpt_dense)
    log_var = Lambda(split, arguments={'start': mu_dim, 'end': mu_dim * 2})(cpt_dense)
    z = reparameterize(mu_dim, batch_size)([mu, log_var])

    es_encoder = Model(inputs=cpt_input, outputs=[mu, log_var, z])
    return es_encoder

# 双时间分量变分编码器
def Duplex_Variational_Encoder(H=10, W=20, in_channel=64, out_channel=64, mu_dim=32, batch_size=8):
    x_input = Input(batch_shape=(batch_size, in_channel, H, W))
    y_input = Input(batch_shape=(batch_size, in_channel, H, W))
    share_input = Concatenate(axis=1)([x_input, y_input])

    share_out = Conv2D(filters=out_channel, kernel_size=(1, 1), padding="same")(share_input)
    share_dense = Flatten()(share_out)
    share_dense = Dense(units=mu_dim * 2)(share_dense)

    mu = Lambda(split, arguments={'start': 0, 'end': mu_dim})(share_dense)
    log_var = Lambda(split, arguments={'start': mu_dim, 'end': mu_dim * 2})(share_dense)
    z = reparameterize(mu_dim, batch_size)([mu, log_var])

    cs_encoder = Model(inputs=[x_input, y_input], outputs=[mu, log_var, z, share_out])
    return cs_encoder

# 共享时间分量变分编码器
def Interactive_Encoder(H=10, W=20, in_channel=64, out_channel=64, mu_dim=32, batch_size=8):
    c_input = Input(batch_shape=(batch_size, in_channel, H, W))
    p_input = Input(batch_shape=(batch_size, in_channel, H, W))
    t_input = Input(batch_shape=(batch_size, in_channel, H, W))
    share_input = Concatenate(axis=1)([c_input, p_input, t_input])

    share_out = Conv2D(filters=out_channel, kernel_size=(1, 1), padding="same")(share_input)
    share_dense = Flatten()(share_out)
    share_dense = Dense(units=mu_dim * 2)(share_dense)

    mu = Lambda(split, arguments={'start': 0, 'end': mu_dim})(share_dense)
    log_var = Lambda(split, arguments={'start': mu_dim, 'end': mu_dim * 2})(share_dense)
    z = reparameterize(mu_dim, batch_size)([mu, log_var])

    cs_encoder = Model(inputs=[c_input, p_input, t_input], outputs=[mu, log_var, z, share_out])
    return cs_encoder

def Reconstructed_Decoder(H=10, W=20, in_channel=32, out_channel=64, batch_size=8):
    cpt_input = Input(batch_shape=(batch_size, in_channel))

    dim = out_channel * H * W
    cpt_out = Dense(units=dim)(cpt_input)
    cpt_out = BatchNormalization()(cpt_out)
    cpt_out = Activation('tanh')(cpt_out)
    cpt_out = Reshape((out_channel, H, W))(cpt_out)

    decoder = Model(inputs=cpt_input, outputs=cpt_out)
    return decoder


def cpt_slice(x, h1, h2):
    return x[:,h1:h2,:,:]


def Disentangle_Network(batch_size=8, H=21, W=12,
                        channel=2, c=3, p=4, t=4,
                        feat_dim=64, conv=64, mu_dim=32,
                        R_N=2, plus=8, rate=2, drop=0, lr=0.0002):

    all_channel = channel * (c + p + t)

    c_dim = channel * c
    p_dim = channel * p
    t_dim = channel * t

    cut0 = int(0)
    cut1 = int(cut0 + c_dim)
    cut2 = int(cut1 + p_dim)
    cut3 = int(cut2 + t_dim)

    cpt_input = Input(shape=(all_channel, H, W))

    c_input = Lambda(cpt_slice, arguments={'h1': cut0, 'h2': cut1})(cpt_input)
    p_input = Lambda(cpt_slice, arguments={'h1': cut1, 'h2': cut2})(cpt_input)
    t_input = Lambda(cpt_slice, arguments={'h1': cut2, 'h2': cut3})(cpt_input)

    # Exclusive encoder to generate the exclusive representation and corresponding distribution
    c_mu, c_log_var, c_z, c_feat = Exclusive_Encoder(H, W, c_dim, feat_dim, mu_dim // 4, batch_size)(c_input)
    p_mu, p_log_var, p_z, p_feat = Exclusive_Encoder(H, W, p_dim, feat_dim, mu_dim // 4, batch_size)(p_input)
    t_mu, t_log_var, t_z, t_feat = Exclusive_Encoder(H, W, t_dim, feat_dim, mu_dim // 4, batch_size)(t_input)


    # Extract the interactive features which are used to simplex and duplex variational encoder, and interactive encoder
    cs_feat = Interactive_Feature_extractor(H, W, c_dim, feat_dim, batch_size)(c_input)
    ps_feat = Interactive_Feature_extractor(H, W, p_dim, feat_dim, batch_size)(p_input)
    ts_feat = Interactive_Feature_extractor(H, W, t_dim, feat_dim, batch_size)(t_input)

    # Simplex Variational Encoder is to calculate the distribution (mu and log_var) of single time sub-series
    cs_mu, cs_log_var, cs_z = Simplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)(cs_feat)
    ps_mu, ps_log_var, ps_z = Simplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)(ps_feat)
    ts_mu, ts_log_var, ts_z = Simplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)(ts_feat)

    # Duplex Variational Encoder is to calculate the distribution (mu and log_var) of paired time sub-series
    cps_mu, cps_log_var, cp_z, _ = Duplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)([cs_feat, ps_feat])
    cts_mu, cts_log_var, ct_z, _ = Duplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)([cs_feat, ts_feat])
    pts_mu, pts_log_var, pt_z, _ = Duplex_Variational_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)([ps_feat, ts_feat])

    # Interactive Encoder is to extract interactive representation and distribution
    s_mu, s_log_var, s_z, cpt_share = Interactive_Encoder(H, W, feat_dim, conv, mu_dim, batch_size)([cs_feat, ps_feat, ts_feat])

    # Reconstructed Decoder is to reconstruct original time sub-series by using the distribution
    c_dec = Reconstructed_Decoder(H, W, mu_dim // 4 + mu_dim, c_dim, batch_size)(Concatenate(axis=1)([c_z, s_z]))
    p_dec = Reconstructed_Decoder(H, W, mu_dim // 4 + mu_dim, p_dim, batch_size)(Concatenate(axis=1)([p_z, s_z]))
    t_dec = Reconstructed_Decoder(H, W, mu_dim // 4 + mu_dim, t_dim, batch_size)(Concatenate(axis=1)([t_z, s_z]))

    # Concatenate the exclusive and interactive representations to capture the spatial dependency via Res_plus proposed by DeepSTN+
    cpt = Concatenate(axis=1)([c_feat, p_feat, t_feat, cpt_share])
    cpt = conv_unit1(conv * 4, conv, drop, H, W)(cpt)

    for i in range(R_N):
        cpt = Res_plus(conv, plus, rate, drop, H, W)(cpt)

    # Prediction
    cpt_out = Activation('relu')(cpt)
    cpt_out = BatchNormalization()(cpt_out)
    cpt_out = Dropout(drop)(cpt_out)
    cpt_out = Conv2D(filters=channel, kernel_size=(1, 1), padding="same")(cpt_out)
    cpt_out = Activation('tanh', name='merge')(cpt_out)

    disentangle_model = Model(inputs=cpt_input, outputs=cpt_out)

    disentangle_model.compile(loss={'merge': loss_function(cut0, cut1, cut2, cut3,
                                                           c_dec, p_dec, t_dec,
                                                           c_mu, p_mu, t_mu,
                                                           cs_mu, ps_mu, ts_mu,
                                                           cps_mu, cts_mu, pts_mu,
                                                           c_log_var, p_log_var, t_log_var,
                                                           cps_log_var, cts_log_var, pts_log_var,
                                                           cs_log_var, ps_log_var, ts_log_var,
                                                           s_mu, s_log_var)},
                              optimizer=Adam(lr), metrics=[metrics.pickup_rmse, metrics.pickup_mae, metrics.pickup_mape, metrics.dropoff_rmse, metrics.dropoff_mae, metrics.dropoff_mape])

    return disentangle_model







