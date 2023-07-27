import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np

class GCNdata(Dataset):
  def __init__(self, users, items, ratings):
    super(GCNdata, self).__init__()
    self.users = users
    self.items = items
    self.ratings = ratings
    self.Graphs = None
    self.n_users = users.max()+1
    self.m_items = items.max()+1
  
  def __len__(self):
    return len(self.ratings)
  
  def __getitem__(self, idx):
    users = self.users[idx]
    items = self.items[idx]
    ratings = self.ratings[idx]
    return users, items, ratings
  
  def getSparseGraphs(self, itemID_parties, args):
    if self.Graphs is None:
      self.Graphs = []
      pos_rating = (self.ratings >= args.thd)
      last_item_idx = 0
      for i in range(args.n_parties):
        all_items = itemID_parties[i]
        this_item_num = len(all_items)
        this_party_idx = np.isin(self.items, all_items)
        pos_users = self.users[pos_rating & this_party_idx]
        pos_items = self.items[pos_rating & this_party_idx]
        user_dim = torch.LongTensor(pos_users)
        item_dim = torch.LongTensor(pos_items)
        item_dim = item_dim - last_item_idx
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        this_graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+this_item_num, self.n_users+this_item_num]))
        dense = this_graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        dense = dense/np.sqrt(self.m_items/this_item_num)
        dense = dense.fill_diagonal_(1)
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        this_graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+this_item_num, self.n_users+this_item_num]))
        this_graph = this_graph.coalesce()
        self.Graphs.append(this_graph)
        last_item_idx += this_item_num
    return self.Graphs


class GATdata(Dataset):
  def __init__(self, users, items, ratings):
    super(GATdata, self).__init__()
    self.users = users
    self.items = items
    self.ratings = ratings
    self.Graphs = None
    self.n_users = users.max()+1
    self.m_items = items.max()+1
  
  def __len__(self):
    return len(self.ratings)
  
  def __getitem__(self, idx):
    users = self.users[idx]
    items = self.items[idx]
    ratings = self.ratings[idx]
    return users, items, ratings
  
  def getSparseGraphs(self, itemID_parties, args):
    if self.Graphs is None:
      self.Graphs = []
      pos_rating = (self.ratings >= args.thd)
      last_item_idx = 0
      for i in range(args.n_parties):
        all_items = itemID_parties[i]
        this_item_num = len(all_items)
        this_party_idx = np.isin(self.items, all_items)
        pos_users = self.users[pos_rating & this_party_idx]
        pos_items = self.items[pos_rating & this_party_idx]
        user_dim = torch.LongTensor(pos_users)
        item_dim = torch.LongTensor(pos_items)
        item_dim = item_dim - last_item_idx
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        this_graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+this_item_num, self.n_users+this_item_num]))
        self.Graphs.append(this_graph)
        last_item_idx += this_item_num
    return self.Graphs

class GGNNdata(Dataset):
    def __init__(self, users, items, ratings):
        super(GGNNdata, self).__init__()
        self.users = users
        self.items = items
        self.ratings = ratings
        self.Graphs = None
        self.n_users = users.max()+1
        self.m_items = items.max()+1
  
    def __len__(self):
        return len(self.ratings)
  
    def __getitem__(self, idx):
        users = self.users[idx]
        items = self.items[idx]
        ratings = self.ratings[idx]
        return users, items, ratings
  
    def getSparseGraphs(self, itemID_parties, args):
        if self.Graphs is None:
            self.Graphs = []
            pos_rating = (self.ratings >= args.thd)
            last_item_idx = 0
            for i in range(args.n_parties):
                all_items = itemID_parties[i]
                this_item_num = len(all_items)
                this_party_idx = np.isin(self.items, all_items)
                pos_users = self.users[pos_rating & this_party_idx]
                pos_items = self.items[pos_rating & this_party_idx]
                user_dim = torch.LongTensor(pos_users)
                item_dim = torch.LongTensor(pos_items)
                item_dim = item_dim - last_item_idx
                first_sub = torch.stack([user_dim, item_dim + self.n_users])
                second_sub = torch.stack([item_dim + self.n_users, user_dim])
                index = torch.cat([first_sub, second_sub], dim=1)
                data = torch.ones(index.size(-1)).int()
                this_graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+this_item_num, self.n_users+this_item_num]))
                dense = this_graph.to_dense()
                D = torch.sum(dense, dim=1).unsqueeze(dim=1).float()
                D[D==0.] = 1.
                dense = dense/D
                index = dense.nonzero()
                data  = dense[dense >= 1e-9]
                assert len(index) == len(data)
                this_graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+this_item_num, self.n_users+this_item_num]))
                this_graph = this_graph.coalesce()
                self.Graphs.append(this_graph)
                last_item_idx += this_item_num
        return self.Graphs