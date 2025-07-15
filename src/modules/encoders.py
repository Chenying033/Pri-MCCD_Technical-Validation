import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        bertconfig = BertConfig()
        self.bertmodel = BertModel(bertconfig)

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)



class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, n_class, dropout=0.0):
        super(SubNet, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        # print(f"SubNet input shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)  # 仅在需要时进行展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"SubNet output shape: {x.shape}")
        return x




class CLUB(nn.Module):
    """
        Compute the Contrastive Log-ratio Upper Bound (CLUB) given a pair of inputs.
        Refer to https://arxiv.org/pdf/2006.12013.pdf and https://github.com/Linear95/CLUB/blob/f3457fc250a5773a6c476d79cda8cb07e1621313/MI_DA/MNISTModel_DANN.py#L233-254

        Args:
            hidden_size(int): embedding size
            activation(int): the activation function in the middle layer of MLP
    """
    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, modal_a, modal_b, sample=False):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]

            Args:
                modal_a (Tensor): x in above equation
                model_b (Tensor): y in above equation
        """
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a) # (bs, hidden_size)
        batch_size = mu.size(0)
        pred = mu

        # pred b using a
        pred_tile = mu.unsqueeze(1).repeat(1, batch_size, 1)   # (bs, bs, emb_size)
        true_tile = pred.unsqueeze(0).repeat(batch_size, 1, 1)      # (bs, bs, emb_size)

        positive = - (mu - modal_b) ** 2 /2./ torch.exp(logvar)
        negative = - torch.mean((true_tile-pred_tile)**2, dim=1)/2./torch.exp(logvar)

        lld = torch.mean(torch.sum(positive, -1))
        bound = torch.mean(torch.sum(positive, -1)-torch.sum(negative, -1))
        return lld, bound

class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)
        batch_size = mu.size(0)

        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive,-1))

        # For Gaussian Distribution Estimation
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos':None, 'neg':None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y)
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None and mem.get('neg', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']
                # Diagonal setting
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                print(neg_y)
                # print(neg_history.shape, neg_y.shape)
                print(pos_all.shape, neg_all.shape)

                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)

                print('neg_all: ', torch.sum(torch.isnan(neg_all)))
                print('mu_neg: ', torch.sum(torch.isnan(mu_neg)))
                sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                a = 17.0795
                print(torch.sum(torch.isnan(sigma_pos)))
                print(torch.sum(torch.isnan(sigma_neg)))
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))
                print(torch.sum(torch.isnan(H)))
                assert False
        return lld, sample_dict, H

class CPC(nn.Module):

    ###############baseline听觉
    # def __init__(self, in_features, out_features):
    #     super(CPC, self).__init__()
    #     # Define the network with a Linear layer
    #     self.net = nn.Sequential(
    #         nn.Linear(in_features, out_features)  # in_features should match the output features of ResNet
    #     )
    #
    # def forward(self, x, y):
    #     # print(f"CPC input x shape: {x.shape}")  # Print the shape of x
    #     # print(f"CPC input y shape: {y.shape}")  # Print the shape of y
    #     x_pred = self.net(y)  # Forward pass through the Linear layer
    #     return x_pred


  #################baseline视觉
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        # print(f'Initializing CPC with x_size={x_size}, y_size={y_size}')
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=self.y_size,  # Ensure this matches y's last dimension
                out_features=self.x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calculate the score"""
        # print(f"CPC x shape: {x.shape}")
        # print(f"CPC y shape: {y.shape}")

        # print(f"Input to Linear layer shape: {y.shape}")
        # print(f"CPC input x shape: {x.shape}")
        # print(f"CPC input y shape: {y.shape}")
        x_pred = self.net(y)    # bs, emb_size
        # print(f"CPC x_pred shape: {x_pred.shape}")

        # Normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        # print(f"Normalized x shape: {x.shape}")
        # print(f"Normalized x_pred shape: {x_pred.shape}")

        pos = torch.sum(x * x_pred, dim=-1)   # bs
        print(f"Positive shape: {pos.shape}")
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        print(f"Negative shape: {neg.shape}")
        nce = -(pos - neg).mean()
        return nce


class Encoder(nn.Module):
    def __init__(self, encoder_type, in_size, hidden_size, out_size, num_layers=1, dropout=0.0, bidirectional=False, kernel_size=3):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type.lower()
        self.out_size = out_size

        if self.encoder_type == 'rnn':
            self.rnn = nn.LSTM(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), out_size)

        elif self.encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), out_size)

        elif self.encoder_type == 'tcn':
            from torch.nn.utils import weight_norm
            layers = []
            num_channels = [hidden_size] * num_layers
            for i in range(num_layers):
                dilation_size = 2 ** i
                in_ch = in_size if i == 0 else hidden_size
                layers += [weight_norm(nn.Conv1d(in_ch, hidden_size, kernel_size,
                                                 padding=(kernel_size - 1) * dilation_size,
                                                 dilation=dilation_size)),
                           nn.ReLU()]
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(hidden_size, out_size)

        elif self.encoder_type == 'mlp':
            layers = []
            input_dim = in_size
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_size
            layers.append(nn.Linear(hidden_size, out_size))
            self.network = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x, lengths=None):
        if self.encoder_type in ['rnn', 'gru']:
            lengths = lengths.cpu().to(torch.int64)
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, h_n = self.rnn(packed_input) if self.encoder_type == 'gru' else self.rnn(packed_input)[0:2]
            if isinstance(h_n, tuple):  # LSTM: (h_n, c_n)
                h_n = h_n[0]
            final_output = h_n[-1]
            return self.fc(final_output)

        elif self.encoder_type == 'tcn':
            x = x.transpose(1, 2)  # (B, D, T)
            out = self.network(x)
            out = out[:, :, -1]  # 取最后时刻
            return self.fc(out)

        elif self.encoder_type == 'mlp':
            x = x.mean(dim=1)  # 简单平均池化
            return self.network(x)


def get_encoder(encoder_type, in_size, hidden_size, out_size, num_layers=1, dropout=0.0, bidirectional=False, kernel_size=3):
    return Encoder(
        encoder_type=encoder_type,
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        kernel_size=kernel_size
    )
