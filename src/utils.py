import numpy as np
import torch

def get_itr_graph(Graphs, n_users, args):
  graphs = []
  for i in range(args.n_parties):
    this_g = Graphs[i].to_dense()
    this_graph = this_g[:n_users, n_users:]
    graphs.append(this_graph)
  return graphs

def sparseProject(w, d, t):
  sparseMat = np.zeros((w*t, d))
  for i in range(t):
    hashIdx = np.random.choice(w, d, replace=True)
    randSigns = np.random.choice([-1, 1], d, replace=True)
    for j in range(w):
      sparseMat[j+w*i] = (hashIdx == j) * randSigns
  return sparseMat / np.sqrt(t)

def metrics(models, dataloader, itemID_parties, n_users, phi, max_rating, args):
    RMSE = np.array([], dtype = np.float32)
    useremb_aggs = torch.zeros(args.n_parties, args.n_layers, n_users, args.latent_dim)
    for layer in range(args.n_layers):
        for i in range(args.n_parties):
            this_model = models[i]
            useremb_aggs[i, layer] = this_model.users_emb_agg[layer].data

    for users, items, ratings in dataloader:
        last_item_idx = 0
        for i in range(args.n_parties):
            all_items = itemID_parties[i]
            this_idx = torch.isin(items, all_items)
            this_idx = this_idx.nonzero(as_tuple=True)[0]
            this_users = users[this_idx]
            this_items = items[this_idx] - last_item_idx
            this_ratings = ratings[this_idx]
            this_model = models[i]
            this_useremb_aggs = 0
            for j in range(args.n_parties):
                if j!=i:
                    this_useremb_aggs += useremb_aggs[j]
            this_useremb_aggs = torch.einsum('pu,huk->hpk', phi, this_useremb_aggs)
            prediction = this_model(this_users, this_items, this_useremb_aggs)
            this_ratings = this_ratings.float()
            prediction = prediction.clamp(min=0.0, max=max_rating)
            SE = (prediction - this_ratings).pow(2)
            RMSE = np.append(RMSE, SE.detach().cpu().numpy())
            last_item_idx += this_model.num_items
    return np.sqrt(RMSE.mean())