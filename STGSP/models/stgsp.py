import torch
import torch.nn as nn
from models.layer.sfe import Conv1ResUnitsConv2
from models.ModelConfiguration import ModelConfigurationTaxiBJ, ModelConfigurationBikeNYC, ModelConfigurationTaxiNYC

class STGSP(nn.Module):
    def __init__(self, data_conf):
        super().__init__()

        # config
        self.dconf = data_conf
        if self.dconf.name == 'BikeNYC':
            self.mconf = ModelConfigurationBikeNYC()
        elif self.dconf.name == 'TaxiNYC':
            self.mconf = ModelConfigurationTaxiNYC()
        elif self.dconf.name == 'TaxiBJ':
            self.mconf = ModelConfigurationTaxiBJ()
        else:
            raise ValueError('The data set does not exist')
        self.mconf.show()

        self.resnns = nn.ModuleList()
        for i in range(self.dconf.len_seq):
            resnn = Conv1ResUnitsConv2(self.dconf, self.mconf)
            self.resnns.append(resnn)

        self.pre_token = nn.Parameter(torch.randn(1, self.mconf.transformer_dmodel))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.mconf.transformer_dmodel, nhead=self.mconf.transformer_nhead, dropout=self.mconf.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.mconf.transformer_nlayers)
        
        self.FC = nn.Linear(in_features=self.mconf.transformer_dmodel*(self.dconf.len_seq+1), out_features=self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w)

        self.model_name = str(type(self).__name__)

    def forward(self, X):
        inputs = torch.split(X, self.dconf.dim_flow, 1)
        
        transformer_inputs = []
        pre_tokens = self.pre_token.repeat(inputs[0].shape[0], 1)
        transformer_inputs.append(pre_tokens)
        for i in range(self.dconf.len_seq):
            X_em = self.resnns[i](inputs[i])
            transformer_inputs.append(X_em)

        transformer_inputs = torch.stack(transformer_inputs, 0)
        transformer_outputs = self.transformer_encoder(transformer_inputs)
        transformer_outputs = transformer_outputs.transpose(0, 1)
        out = torch.flatten(transformer_outputs, 1)
        out = self.FC(out)

        main_out = out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        main_out = torch.tanh(main_out)
        return main_out

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")
