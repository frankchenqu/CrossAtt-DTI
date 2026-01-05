# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 19:40
@author: LiFan Chen
@Filename: model_glu.py
@Software: PyCharm
"""
# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/7 13:40
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import main_glu

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        if len(query.shape) > len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        Q, K = Q.cpu(), K.cpu()
        del Q, K
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        return self.fc(
            torch.matmul(self.do(F.softmax(energy, dim=-1)), V).permute(0, 2, 1, 3).contiguous().view(bsz, -1,
                                                                                                      self.n_heads * (
                                                                                                                  self.hid_dim // self.n_heads)))
class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):

        conv_input = self.fc(protein)

        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.do(F.relu(self.fc_1(x)))

        x = self.fc_2(x)

        x = x.permute(0, 2, 1)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)           # hid_dim = 64，n_heads = 8
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):

        trg1 = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, src, src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))

        src1 = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, trg, trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))
        trg,src= trg.cpu(),src.cpu()
        del trg,src, trg_mask, src_mask

        m1 = torch.mean(trg1, 1)
        trg1 = torch.unsqueeze(m1, 1)
        m2 = torch.mean(src1, 1)
        src1 = torch.unsqueeze(m2, 1)

        return trg1,src1



# IMT

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        for layer in self.layers:
            trg,src = layer(trg, src,trg_mask,src_mask)
        del trg_mask,src_mask
        return trg,src


# GAT
class GraphAttentionLayer(nn.Module):
    """
    实现GAT中的注意力机制的基本模块。
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.matmul(h, self.W)

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        N = Wh.size()[0]       # 节点个数
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)    # 拼接两个重复矩阵
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):

    def __init__(self, atom_dim, hid_dim, gat_heads, dropout, alpha, n_layers,device):
        super(GAT, self).__init__()

        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim) for _ in range(n_layers)])
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(n_layers)])


        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.atom_dim = atom_dim
        self.attentions = [GraphAttentionLayer(atom_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(gat_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hid_dim * gat_heads, atom_dim, dropout=dropout, alpha=alpha, concat=False)  # 输出层的注意力头


    def forward(self, x, adj,n_layers):


        for i in range(n_layers):
            h = torch.relu(self.W_gnn[i](x))
            size = h.size()[0]
            N = h.size()[1]

            h1 = h.repeat(1,1, N)
            h2 = h1.view(size, N * N, -1)
            h3 = h.repeat(1, N, 1)
            h4 = torch.cat([h2,h3],dim=2)
            a_input = h4.view(size, N, -1, 2 * self.atom_dim)

            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, h)
            x = x+h_prime   # (1,78,34)
        return torch.unsqueeze(torch.mean(x, 1), 1)

# BERT



# CNN-protein
class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
           # nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
           # nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
          #  nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
          #  nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
          #  nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
          #  nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
          #  nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
         #  nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  #[bs, hid_dim, seq_len]
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)  #[bs, hid_dim*3, seq_len]
        #features = features.max(dim=-1)[0]  #[bs, hid_dim*3]
        features = features.permute([0,2,1])  #[bs, hid_dim*3, seq_len] ——>[bs, seq_len, hid_dim*3]
        features = self.conv(features)     # [bs, seq_len, hid_dim*3] ——>[bs, seq_len, hid_dim]
        return torch.unsqueeze(torch.mean(features, 1), 1)  # [bs, seq_len, hid_dim] ——>[bs, hid_dim] ——>[bs, 1,hid_dim]


class InteractionModel(nn.Module):
    def __init__(self,hid_dim, n_heads):
        super(InteractionModel, self).__init__()
        # self.compound_embedding = nn.Linear(compound_feature_size, hidden_size)
        # self.protein_embedding = nn.Linear(protein_feature_size, hidden_size)
        self.compound_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.protein_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.compound_fc = nn.Linear(hid_dim, hid_dim)
        self.protein_fc = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()

        self.hid_dim = hid_dim


    def forward(self, compound_features, protein_features):
        compound_embedded = self.activation(compound_features)
        protein_embedded = self.activation(protein_features)

        compound_embedded = compound_embedded.permute(1, 0, 2)
        protein_embedded = protein_embedded.permute(1, 0, 2)


        compound_attention_output, _ = self.compound_attention(compound_embedded, compound_embedded,
                                                               compound_embedded)
        protein_attention_output, _ = self.protein_attention(protein_embedded, protein_embedded, protein_embedded)

        compound_attention_output = compound_attention_output.permute(1, 0, 2)
        protein_attention_output = protein_attention_output.permute(1, 0, 2)

        compound_output = self.activation(self.compound_fc(compound_attention_output))
        protein_output = self.activation(self.protein_fc(protein_attention_output))

        com_att = torch.unsqueeze(torch.mean(compound_output,1),1)
        pro_att = torch.unsqueeze(torch.mean(protein_output,1),1)
        return com_att,pro_att




# NTN  (tensor_network)

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self,k_feature,hid_dim,k_dim):
        super(TensorNetworkModule, self).__init__()
        self.k_feature = k_feature
        self.hid_dim = hid_dim
        self.k_dim = k_dim

        self.setup_weights()
        self.init_parameters()

        self.fc1 = nn.Linear(hid_dim,k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)


    def setup_weights(self):
        """
        Defining weights.  k_feature = args.filters_3   args.tensor_neurons = k_dim
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.k_feature, self.k_feature, self.k_dim
            )
        )                                                             # (16,16,16)
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.k_dim, 2 * self.k_feature)
        )                                                             # (16,32)
        self.bias = torch.nn.Parameter(torch.Tensor(self.k_dim, 1))   # (16,1)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.   com_att
        :param embedding_2: Result of the 2nd embedding after attention.   pro_att
        :return scores: A similarity score vector.
        """
        embedding_1 = torch.squeeze(embedding_1, dim=1)   # (1,1,64)——>(1,64)   (batch_size,1,64)——>(batch_size,64)
        embedding_1 = self.fc1(embedding_1)  # (1,64)——>(1,16)
        embedding_2 = torch.squeeze(embedding_2, dim=1)
        embedding_2 = self.fc1(embedding_2)

        batch_size = len(embedding_1)

        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.k_feature, -1)
        )
        scoring = scoring.view(batch_size, self.k_feature, -1).permute([0, 2, 1])
        # print(scoring.shape)
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.k_feature, 1)
        ).view(batch_size, -1)
        # print(scoring.shape)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        # print(combined_representation.shape)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )
        # print(block_scoring.shape)
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        # print(scores.shape)    # (1,16)——(batch_size,16)
        scores = torch.unsqueeze(scores,1)  # (1,16) ——> (1,1,16)
        scores = self.fc2(scores)    # (1,1,16) ——> (1,1,64)
        return scores




class Predictor(nn.Module):
    def __init__(self,gat,cnn, decoder,inter_att,tensor_network, device,n_fingerprint,n_layers,atom_dim=34):
        super().__init__()

        self.embed_fingerprint = nn.Embedding(n_fingerprint, atom_dim)
        self.gat = gat
        #self.Bert = bert
        self.prot_textcnn = cnn
        self.inter_att = inter_att
        self.tensor_network = tensor_network
        # self.embed_word = nn.Embedding(n_word, atom_dim)

        # self.encoder = encoder
        self.n_layers = n_layers
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(34, 34))
        self.init_weight()


        self.protein_dim = 100
        self.hid_dim = 64
        self.atom_dim = 34
        self.fc1 = nn.Linear(self.protein_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.atom_dim, self.hid_dim)
        self.fc3 = nn.Linear(1, self.protein_dim)

        self.W_attention = nn.Linear(self.hid_dim, self.hid_dim)



        self.out = nn.Sequential(
            nn.Linear(self.hid_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )



    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)




    def forward(self, compound1, compound2,adj, protein1,protein2,n_layers):

        protein1 = torch.unsqueeze(protein1, dim=0)    # (1,478,100)    # protein1 =[ batch size=1,protein len, protein_dim]
        protein1 = self.fc1(protein1)                  # (1,478,64)     # protein1 =[ batch size=1,protein len, hid_dim]


        compound1 = torch.unsqueeze(compound1, dim=0)  # (1,31,34)       # compound1 = [batch size=1 ,atom_num, atom_dim]
        compound1 = self.fc2(compound1)                # (1,31,64)       # compound1 = [batch size=1 ,atom_num, hid_dim]

        protein1_c, compound1_p = self.decoder(protein1, compound1)

        compound_vectors = self.embed_fingerprint(compound2)            # compound_vectors:(47,34)
        compound2 = torch.unsqueeze(compound_vectors, dim=0)   # 1,47,34
        compound2 = self.gat(compound2, torch.unsqueeze(adj,0),n_layers)  # 1,47,34 ——>1,1,34
        compound2 = self.fc2(compound2)                                 # 1,1,34 ——> 1,1,64


        protein2 = self.prot_textcnn(protein1)


        out_fc = torch.cat((compound2, protein2,compound1_p, protein1_c), 2)
        out = self.out(out_fc)    # out = [batch size, 2]
        out = torch.squeeze(out, dim=0)
        return out



    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]

        compound1,compound2, adj, protein1, protein2 = inputs
        Loss = nn.CrossEntropyLoss()

        n_layers =3
        if train:

            predicted_interaction = self.forward(compound1,compound2, adj, protein1,protein2,n_layers)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            predicted_interaction = self.forward(compound1,compound2, adj, protein1,protein2,n_layers)
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction,1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0,1]
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        for data in dataset:
            i = i+1
            loss = self.model(data)
            loss = loss / self.batch
            loss.backward()
            if i % self.batch  == 0 or  i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall
        # return S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
