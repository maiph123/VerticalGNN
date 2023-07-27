import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod

class base_mod(nn.Module):
    def __init__(self, dataset, graph, itemID_parties, party_num, args):
        super(GAT, self).__init__()
        self.dataset = dataset
        self.Graph = graph
        self.latent_dim = args.latent_dim
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob
        self.A_split = args.A_split
        self.pretrain = args.pretrain
        self.dropout = args.dropout 
        self.alpha = args.alpha
        self.thd = args.thd

        if args.act_function == 'sigmoid':
          self.f = nn.Sigmoid()
        elif args.act_function == 'relu':
          self.f = nn.LeakyReLU()
        elif args.act_function == 'tanh':
          self.f = nn.Tanh()
        torch.manual_seed(12345)
        W_data = torch.normal(mean=0, std=1, size=(args.latent_dim, args.latent_dim))
        layer_agg_weight = torch.ones(args.n_layers+1)/(args.n_layers+1)
        self.layer_agg_weight = nn.parameter.Parameter(data=layer_agg_weight, requires_grad=True)
        
    @abstractmethod
    def __init_weight(self, itemID_parties, party_num, args):
        pass

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        g = g.to_dense()
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    @abstractmethod
    def computer(self, missing_user_aggs, phi_inv, args):
        pass

    def sqr_loss(self, ratings, prediction):
        itemEmb0 = self.embedding_item.weight
        reg_loss = itemEmb0.norm(2).pow(2)/self.num_items
        loss = (prediction - ratings).pow(2).sum()
        loss = loss + reg_loss
        return loss
       
    def forward(self, users, items, missing_user_aggs=None):
        # compute embedding
        all_users, all_items = self.computer(missing_user_aggs)
        
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        inner_pro = torch.mul(users_emb, items_emb)
        ratings = torch.sum(inner_pro, dim=1)
        return ratings

class GAT(base_mod):
    def __init__(self, dataset, graph, itemID_parties, party_num, args):
        super(GAT, self).__init__()
        self.a = nn.Parameter(torch.ones(size=(2*args.latent_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.feat_transform = nn.ParameterList([nn.Parameter(W_data) for i in range(args.n_layers)])
        self.__init_weight(itemID_parties, party_num, args)

    def __init_weight(self, itemID_parties, party_num, args):
        self.num_users  = self.dataset.n_users
        self.num_items  = len(itemID_parties[party_num])
        self.num_all_items  = self.dataset.m_items
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.pretrain:   
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_user.weight))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_item.weight))
        else:
            torch.manual_seed(12345)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            torch.manual_seed(np.random.randint(10000))
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # compute aggregated user embeddings
        g_droped = self.Graph.to_dense()
        g_droped = g_droped.fill_diagonal_(1)
        self.users_emb_agg = [0 for i in range(self.n_layers)]
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            attention = self._prepare_attentional_mechanism_input(all_emb, layer, g_droped)
            self.users_emb_agg[layer] = torch.mm(attention[:self.num_users, self.num_users:], all_emb[self.num_users:])
            all_emb = torch.mm(attention, all_emb)
            W = self.feat_transform[layer]
            all_emb = torch.mm(all_emb, W)
            all_emb = self.f(all_emb)
            embs.append(all_emb)
        
        if args.device == 'cuda':
          self.Graph = self.Graph.cuda()
   
    def _prepare_attentional_mechanism_input(self, h, layer, g_droped):
        W = self.feat_transform[layer]
        Wh = torch.mm(h, W) # h.shape: (N+M, latent_dim), Wh.shape: (N+M, latent_dim)
        Wh1 = torch.matmul(Wh, self.a[:self.latent_dim, :]) # Wh1.shape: (N+M, 1)
        Wh2 = torch.matmul(Wh, self.a[self.latent_dim:, :]) # Wh2.shape: (N+M, 1)
        # broadcast add
        e = Wh1 + Wh2.T # e.shape: (N+M, N+M)
        e = self.leakyrelu(e)
        e = torch.exp(e)
        attention = g_droped * e
        attention_sum_u, attention_sum_i = torch.split(attention, [self.num_users, self.num_items])
        # user_item_sim: N*M matrix
        attention_sum_uu, attention_sum_ui = torch.split(attention_sum_u, [self.num_users, self.num_items], dim=1)
        attention_sum_ui = attention_sum_ui * self.num_all_items / self.num_items
        attention_sum_u = torch.sum(attention_sum_ui, dim=1) + torch.sum(attention_sum_uu, dim=1)
        attention_sum_i = torch.sum(attention_sum_i, dim=1)
        attention_sum = torch.cat((attention_sum_u, attention_sum_i)).unsqueeze(dim=1)
        attention = attention / attention_sum
        return attention

    def computer(self, missing_user_aggs, phi_inv, args):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        if missing_user_aggs != None:
            missing_user_aggs = torch.einsum('up,hpk->huk', phi_inv, missing_user_aggs)
        else:
            missing_user_aggs = [0 for layer in range(self.n_layers)]

        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph.to_dense()
        else:
            g_droped = self.Graph.to_dense()
        g_droped = g_droped.fill_diagonal_(1)
        missing_aggs_append = torch.zeros(self.num_items, self.latent_dim)
        if args.device == 'cuda':
          missing_aggs_append = missing_aggs_append.cuda()

        for layer in range(self.n_layers):
            this_missing_aggs = missing_user_aggs[layer]
            if args.device == 'cuda':
              this_missing_aggs = this_missing_aggs.cuda()
            attention = self._prepare_attentional_mechanism_input(all_emb, layer, g_droped)
            self.users_emb_agg[layer] = torch.mm(attention[:self.num_users, self.num_users:], all_emb[self.num_users:])
            all_emb = torch.mm(attention, all_emb)
            this_missing_aggs = torch.cat((this_missing_aggs, missing_aggs_append))
            all_emb = all_emb + this_missing_aggs
            W = self.feat_transform[layer]
            all_emb = torch.mm(all_emb, W)
            all_emb = self.f(all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=0)
        embs = torch.tensordot(embs, self.layer_agg_weight, dims=([0],[0]))
        users, items = torch.split(embs, [self.num_users, self.num_items])
        return users, items

class GCN(base_mod):
    def __init__(self, dataset, graph, itemID_parties, party_num, args):
        super(GCN, self).__init__()
        self.feat_transform = nn.ModuleList([nn.Linear(args.latent_dim, args.latent_dim, bias=False) for i in range(args.n_layers)])
        self.feat_transform.apply(self.feat_init_weights)
        self.__init_weight(itemID_parties, party_num, args)

    def __init_weight(self, itemID_parties, party_num, args):
        self.num_users  = self.dataset.n_users
        self.num_items  = len(itemID_parties[party_num])
        self.num_all_items  = self.dataset.m_items
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.pretrain:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_user.weight))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_item.weight))
        else:
            torch.manual_seed(12345)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            torch.manual_seed(np.random.randint(10000))
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        # compute aggregated user embeddings
        self.users_emb_agg = [0 for i in range(self.n_layers)]
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        embs_user = [users_emb]
        embs_item = [items_emb]
        for layer in range(self.n_layers):
            this_transform = self.feat_transform[layer]
            this_users_emb = embs_user[-1]
            this_items_emb = embs_item[-1]
            # update item embeddings
            update_items_emb = torch.mm(self.Graph.t(), this_users_emb) + this_items_emb
            update_items_emb = this_transform(update_items_emb)
            update_items_emb = self.f(update_items_emb)
            embs_item.append(update_items_emb)
            # update user embeddings
            self.users_emb_agg[layer] = torch.mm(self.Graph, this_items_emb)
            update_users_emb = self.users_emb_agg[layer] + this_users_emb
            update_users_emb = this_transform(update_users_emb)
            update_users_emb = self.f(update_users_emb)
            embs_user.append(update_users_emb)

    def feat_init_weights(self, m):
        if isinstance(m, nn.Linear):
          torch.manual_seed(12345)
          nn.init.xavier_normal_(m.weight)
    
    def computer(self, missing_user_aggs, phi_inv, cor_ratio, args):
        """
        propagate methods for lightGCN
        """  
        # missing_user_aggs: aggregated user embeddings for each layer     
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        embs_user = [users_emb]
        embs_item = [items_emb]
        if missing_user_aggs != None:
            temp = []
            for j in range(args.n_layers):
                temp.append(torch.mm(phi_inv, missing_user_aggs[j]) * cor_ratio)
            missing_user_aggs = torch.stack(temp)
        else:
            missing_user_aggs = [0 for layer in range(self.n_layers)]

        for layer in range(self.n_layers):
            this_transform = self.feat_transform[layer]
            this_users_emb = embs_user[-1]
            this_items_emb = embs_item[-1]
            this_missing_aggs = missing_user_aggs[layer]
            # update item embeddings
            update_items_emb = torch.mm(self.Graph.t(), this_users_emb) + this_items_emb
            update_items_emb = this_transform(update_items_emb)
            update_items_emb = self.f(update_items_emb)
            embs_item.append(update_items_emb)
            # update user embeddings
            self.users_emb_agg[layer] = torch.mm(self.Graph, this_items_emb)
            update_users_emb = self.users_emb_agg[layer] + this_missing_aggs + this_users_emb
            update_users_emb = this_transform(update_users_emb)
            update_users_emb = self.f(update_users_emb)
            embs_user.append(update_users_emb)
        embs_item = torch.stack(embs_item, dim=0)
        embs_user = torch.stack(embs_user, dim=0)
        items = torch.tensordot(embs_item, self.layer_agg_weight, dims=([0],[0]))
        users = torch.tensordot(embs_user, self.layer_agg_weight, dims=([0],[0]))
        return users, items

class GGNN(nn.Module):
    def __init__(self, dataset, graph, itemID_parties, party_num, args):
        super(GGNN, self).__init__()
        self.tanh = nn.Tanh()
        torch.manual_seed(12345)
        data = torch.randn((args.latent_dim, args.latent_dim))
        self.W = nn.Parameter(data)
        self.Wr = nn.Parameter(data)
        self.Wz = nn.Parameter(data)
        self.U = nn.Parameter(data)
        self.Ur = nn.Parameter(data)
        self.Uz = nn.Parameter(data)
        self.__init_weight(itemID_parties, party_num, args)

    def __init_weight(self, itemID_parties, party_num, args):
        self.num_users  = self.dataset.n_users
        self.num_items  = len(itemID_parties[party_num])
        self.num_all_items  = self.dataset.m_items
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.pretrain:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_user.weight))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.pretrain.embedding_item.weight))
        else:
            torch.manual_seed(12345)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            torch.manual_seed(np.random.randint(10000))
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        # compute aggregated user embeddings
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        g_droped = self.Graph.to_dense()
        self.users_emb_agg = [torch.zeros((self.num_users, self.latent_dim)) for i in range(self.n_layers)]
        for layer in range(self.n_layers):
            emb_aggs = torch.mm(g_droped, all_emb)
            self.users_emb_agg[layer] = emb_aggs[:self.num_users, :]
            all_emb = self.GRU(all_emb, emb_aggs)

        if args.device == 'cuda':
          self.Graph = self.Graph.cuda()
    
    def GRU(self, embs, emb_aggs):
        # Gated Recurrent Unit
        Z = torch.mm(emb_aggs, self.Wz) + torch.mm(embs, self.Uz)
        Z = self.f(Z)
        R = torch.mm(emb_aggs, self.Wr) + torch.mm(embs, self.Ur)
        R = self.f(R)
        e_tilde = torch.mm(emb_aggs, self.W) + R * torch.mm(embs, self.U)
        e_tilde = self.tanh(e_tilde)
        embs_update = (1-Z) * embs + Z * e_tilde
        return embs_update
    
    def computer(self, missing_user_aggs, phi_inv, args):
        """
        propagate methods for lightGCN
        """  
        # missing_user_aggs: aggregated user embeddings for each layer     
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if missing_user_aggs != None:
            missing_user_aggs = torch.einsum('up,hpk->huk', phi_inv, missing_user_aggs)
            # pass
        else:
            missing_user_aggs = [0 for layer in range(self.n_layers)]

        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph.to_dense()
        else:
            g_droped = self.Graph.to_dense()

        for layer in range(self.n_layers):
            this_missing_aggs = missing_user_aggs[layer]
            emb_aggs = torch.mm(g_droped, all_emb)
            user_emb_aggs, item_emb_aggs = torch.split(emb_aggs, [self.num_users, self.num_items])
            user_emb_aggs = user_emb_aggs*self.num_items/self.num_all_items +\
                      this_missing_aggs*(self.num_all_items-self.num_items)/self.num_all_items
            emb_aggs = torch.cat([user_emb_aggs, item_emb_aggs])
            all_emb = self.GRU(all_emb, emb_aggs)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=0)
        embs = torch.tensordot(embs, self.layer_agg_weight, dims=([0],[0]))
        users, items = torch.split(embs, [self.num_users, self.num_items])
        return users, items