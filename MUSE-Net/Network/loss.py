import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error, mean_absolute_error

def reconstruction_loss(origin, recon):
    return mean_absolute_error(origin, recon)
    # return K.mean(mean_absolute_error(origin, recon))


def KLD_loss_v1(mu, log_var):
    kld_loss = K.mean(-0.5 * K.sum(1 + log_var - mu ** 2 - K.exp(log_var), axis=1), axis=0)
    return kld_loss


def KLD_loss(mu0, log_var0, mu1, log_var1):
    kld_loss = K.mean(-0.5 * K.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + K.exp(log_var0) / K.exp(log_var1)), axis=1), axis=0)
    return kld_loss


def loss_function(cut0, cut1, cut2, cut3,
                  c_dec, p_dec, t_dec,
                  c_mu, p_mu, t_mu,
                  cs_mu, ps_mu, ts_mu,
                  cps_mu, cts_mu, pts_mu,
                  c_log_var, p_log_var, t_log_var,
                  cps_log_var, cts_log_var, pts_log_var,
                  cs_log_var, ps_log_var, ts_log_var,
                  s_mu, s_log_var):
    def loss(y_true, y_pred):

        y_true = K.permute_dimensions(y_true, (0, 2, 3, 1))
        y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        c = K.permute_dimensions(c_dec, (0, 2, 3, 1))
        p = K.permute_dimensions(p_dec, (0, 2, 3, 1))
        t = K.permute_dimensions(t_dec, (0, 2, 3, 1))

        y_label = y_true[:, :, :, :2]
        c_data = y_true[:, :, :, 2 + cut0:2 + cut1]
        p_data = y_true[:, :, :, 2 + cut1:2 + cut2]
        t_data = y_true[:, :, :, 2 + cut2:]

        # y_label = y_true[:, :2, :, :]
        # c_data = y_true[:, 2+cut0:2+cut1, :, :]
        # p_data = y_true[:, 2+cut1:2+cut2, :, :]
        # t_data = y_true[:, 2+cut2:, :, :]

        pred_weight = 1
        rec_weight = 1
        X_kld_weight = 0.02
        S_kld_weight = 0.02
        inter_kld_weight = 1
        reg_coeff = 0.0001

        pred_loss = pred_weight * reconstruction_loss(y_pred, y_label)

        recon_loss = rec_weight * (reconstruction_loss(c_data, c) +
                                   reconstruction_loss(p_data, p) +
                                   reconstruction_loss(t_data, t))

        # recon_loss = rec_weight * (K.mean(reconstruction_loss(c_data, c_dec)) +
        #                            K.mean(reconstruction_loss(p_data, p_dec)) +
        #                            K.mean(reconstruction_loss(t_data, t_dec)))

        kl_C_loss = KLD_loss_v1(c_mu, c_log_var)
        kl_P_loss = KLD_loss_v1(p_mu, p_log_var)
        kl_T_loss = KLD_loss_v1(t_mu, t_log_var)

        kl_S_loss = KLD_loss_v1(s_mu, s_log_var)

        kl_inter_CPS_loss = KLD_loss(s_mu, s_log_var, cps_mu, cps_log_var)
        kl_inter_CTS_loss = KLD_loss(s_mu, s_log_var, cts_mu, cts_log_var)
        kl_inter_PTS_loss = KLD_loss(s_mu, s_log_var, pts_mu, pts_log_var)

        kl_inter_CS_loss = KLD_loss(cps_mu, cps_log_var, cs_mu, cs_log_var) + KLD_loss(cts_mu, cts_log_var, cs_mu, cs_log_var)
        kl_inter_PS_loss = KLD_loss(cps_mu, cps_log_var, ps_mu, ps_log_var) + KLD_loss(pts_mu, pts_log_var, cs_mu, cs_log_var)
        kl_inter_TS_loss = KLD_loss(cts_mu, cts_log_var, ts_mu, ts_log_var) + KLD_loss(pts_mu, pts_log_var, cs_mu, cs_log_var)

        disentangle_loss = reg_coeff * recon_loss + \
                           reg_coeff * X_kld_weight * (kl_C_loss + kl_P_loss + kl_T_loss) + \
                           reg_coeff * S_kld_weight * kl_S_loss + \
                           inter_kld_weight * (kl_inter_CS_loss + kl_inter_PS_loss + kl_inter_TS_loss) - \
                           reg_coeff * inter_kld_weight * (kl_inter_CPS_loss + kl_inter_CTS_loss + kl_inter_PTS_loss)

        total_loss = pred_loss + disentangle_loss
        return total_loss

    return loss