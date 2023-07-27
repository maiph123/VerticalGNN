from load_data import *
from model import *
from train import *
from utils import *
import pandas as pd
import torch.optim as optim
import time
import argparse
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=6, help="dimension of latent factors")
    parser.add_argument('--n_layers', type=int, default=2, help="number of layers")
    parser.add_argument('--keep_prob', type=float, default=1, help="probability of keeping the nodes")
    parser.add_argument('--eps', type=float, default=1e-10, help="epsilon")
    parser.add_argument('--A_split', type=bool, default=False, help="whether to split the adjacency matrix")
    parser.add_argument('--pretrain', type=lambda x: None if x == "None" else x, default="None", help="whether to use pretrained model")
    parser.add_argument('--dropout', type=bool, default=True, help="whether to dropout nodes")
    parser.add_argument('--thd', type=float, default=6, help="threshold to be treated as neighbor")
    parser.add_argument('--act_function', type=str, default='sigmoid', help="activation function")
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=5000, help="batch size")
    parser.add_argument('--n_parties', type=int, default=10, help="number of parties")
    parser.add_argument('--clip', type=float, default=3, help="maximum absolute value of gradients")
    parser.add_argument('--device', type=str, default='cuda', help="gpu or cpu device")
    parser.add_argument('--ptcp', type=int, default=10, help="number of participated clients each round")
    parser.add_argument('--dataset', type=str, default="movielens", help="number of participated clients each round", 
                        choices=["movielens", "bookcrossing"])
    parser.add_argument('--dim_red', type=int, default=5, help="dimension reduction ratio")
    parser.add_argument('--model', type=str, default="GCN", help="GNN model", choices=["GCN", "GAT", "GGNN"])
    args = parser.parse_args()
    
    if args.dataset == "movielens":
        data = pd.read_csv('/root/ratings.csv')
    else:
        data = pd.read_csv('/root/Book-Ratings.csv')
        top_users = data.groupby('User-ID')['Book-Rating'].count()
        top_users = top_users.sort_values(ascending=False)[:6000].index
        data = data[data['User-ID'].isin(top_users)]
        top_items = data.groupby('ISBN')['Book-Rating'].count()
        top_items = top_items.sort_values(ascending=False)[:3000].index
        data = data[data['ISBN'].isin(top_items)]
        data.columns = ['UserID', 'MovieID', 'Rating']
    
    userIDs = data.UserID.unique()
    itemIDs = data.MovieID.unique()

    userid2encode = {}
    for i, userid in enumerate(userIDs):
      userid2encode[userid] = i

    itemid2encode = {}
    for i, itemid in enumerate(itemIDs):
      itemid2encode[itemid] = i

    # number of parties
    n_items = max(itemid2encode.values()) + 1
    items_per_party = n_items//args.n_parties
    itemID_parties = {}
    start = 0
    for i in range(args.n_parties):
      if i == args.n_parties-1:
        end = n_items
      else:
        end = start + items_per_party
      itemID_parties[i] = torch.arange(start, end)
      start += items_per_party
    
    data['UserID'] = data['UserID'].apply(lambda x: userid2encode[x])
    data['MovieID'] = data['MovieID'].apply(lambda x: itemid2encode[x])
    data = data.sample(frac=1).reset_index(drop=True)
    max_rating = data.Rating.max()

    train_pct = 0.8
    n_trains = int(len(data)*train_pct)
    train_data = data.loc[:n_trains]
    test_data = data.loc[n_trains:]

    train_users = train_data.UserID.values
    train_items = train_data.MovieID.values
    train_ratings = train_data.Rating.values

    test_users = test_data.UserID.values
    test_items = test_data.MovieID.values
    test_ratings = test_data.Rating.values
    if args.model == "GCN":
        train_dataset = GCNdata(train_users, train_items, train_ratings)
        test_dataset = GCNdata(test_users, test_items, test_ratings)
    elif args.model == "GAT":
        train_dataset = GATdata(train_users, train_items, train_ratings)
        test_dataset = GATdata(test_users, test_items, test_ratings)
    else:
        train_dataset = GGNNdata(train_users, train_items, train_ratings)
        test_dataset = GGNNdata(test_users, test_items, test_ratings)

    train_loader = DataLoader(train_dataset, drop_last=True,
          batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset,
          batch_size=args.batch_size, shuffle=False, num_workers=0)

    dataset = train_dataset

    p = dataset.n_users//args.dim_red
    w = dataset.n_users//args.dim_red
    t = p//w

    phi = sparseProject(w, dataset.n_users, t)
    phi_inv = phi.T
    phi = torch.from_numpy(phi).float()
    phi_inv = torch.from_numpy(phi_inv).float()

    Graphs = train_dataset.getSparseGraphs(itemID_parties, args.thd)
    if args.model == "GCN":
       itr_graphs = get_itr_graph(Graphs)
    models = []
    optimizers = []
    for i in range(args.n_parties):
        if args.model == "GCN":
            this_graph = itr_graphs[i]
            model = GCN(dataset, this_graph, itemID_parties, i, args)
        elif args.model == "GAT":
            this_graph = Graphs[i]
            model = GAT(dataset, this_graph, itemID_parties, i, args)
        else:
            this_graph = Graphs[i]
            model = GGNN(dataset, this_graph, itemID_parties, i, args)
        if args.device == 'cuda':
            model = model.cuda()
        models.append(model)
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
        optimizers.append(optimizer)

    if args.model == "GCN":
        train_dataset = GCNdata(train_users, train_items, train_ratings)
        test_dataset = GCNdata(test_users, test_items, test_ratings)
    elif args.model == "GAT":
        train_dataset = GATdata(train_users, train_items, train_ratings)
        test_dataset = GATdata(test_users, test_items, test_ratings)
    else:
       train_GAT(train_loader, test_loader, models, itemID_parties, optimizers, dataset, phi, args)