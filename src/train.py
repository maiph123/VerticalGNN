import torch
import numpy as np
import time
from utils import metrics

def train_GAT(train_loader, test_loader, models, itemID_parties, optimizers, dataset, phi, args):
        count, best_rmse = 0, 100
        for epoch in range(args.epochs):
          start_time = time.time()
          for users, items, ratings in train_loader:
            selected_parties = np.random.choice(args.n_parties, args.n_ptcp, replace=False)
            # compute gradients
            user_emb_grad = 0
            layer_agg_grad = 0
            a_grad = 0
            feat_transform_grad = {i:0 for i in range(args.n_layers)}
            useremb_aggs = torch.zeros(args.n_parties, args.n_layers, dataset.n_users, args.latent_dim)
            for layer in range(args.n_layers):
              for i in selected_parties:
                this_model = models[i]
                useremb_aggs[i, layer] = this_model.users_emb_agg[layer].data
            
            num_item_choose = 0
            for i in selected_parties:
                num_item_choose += len(itemID_parties[i])
            cor_ratio = dataset.m_items/num_item_choose
            
            for i in selected_parties:
              last_item_idx = 0
              for j in range(i):
                last_item_idx += len(itemID_parties[j])
              all_items = itemID_parties[i]
              this_idx = torch.isin(items, all_items)
              this_idx = this_idx.nonzero(as_tuple=True)[0]
              this_users = users[this_idx]
              this_items = items[this_idx] - last_item_idx
              this_ratings = ratings[this_idx]
              if args.device == 'cuda':
                this_users = this_users.cuda()
                this_items = this_items.cuda()
                this_ratings = this_ratings.cuda()

              this_model = models[i]
              this_model.train()
              this_model.zero_grad()
              this_useremb_aggs = 0
              for j in selected_parties:
                if j!=i:
                  this_useremb_aggs += useremb_aggs[j]
              this_useremb_aggs = torch.einsum('pu,huk->hpk', phi, this_useremb_aggs)
              prediction = this_model(this_users, this_items, this_useremb_aggs)
              loss = this_model.sqr_loss(this_ratings, prediction)
              loss.backward()
              # obtain user embedding gradients
              this_user_emb_grad = this_model.embedding_user.weight.grad
              this_user_emb_grad = torch.clamp(this_user_emb_grad, -0.5, 0.5)
              random_neg = np.random.uniform(low=-args.clip, high=0, size=this_user_emb_grad.shape)
              random_pos = np.random.uniform(low=0, high=args.clip, size=this_user_emb_grad.shape)
              random_neg = torch.from_numpy(random_neg).float()
              random_pos = torch.from_numpy(random_pos).float()
              this_user_emb_grad = (this_user_emb_grad.cpu() > random_pos) * args.clip + (this_user_emb_grad.cpu() < random_neg) * (-args.clip)
              this_user_emb_grad = this_user_emb_grad.float()
              if args.device == 'cuda':
                  this_user_emb_grad = this_user_emb_grad.cuda()
              user_emb_grad += this_user_emb_grad
              # obtain feature transform gradients
              for layer in range(args.n_layers):
                this_feat_transform = this_model.feat_transform[layer].grad
                this_feat_transform = torch.clamp(this_feat_transform, -0.5, 0.5)
                random_neg = np.random.uniform(low=-args.clip, high=0, size=this_feat_transform.shape)
                random_pos = np.random.uniform(low=0, high=args.clip, size=this_feat_transform.shape)
                random_neg = torch.from_numpy(random_neg).float()
                random_pos = torch.from_numpy(random_pos).float()
                this_feat_transform = (this_feat_transform.cpu() > random_pos) * clip + (this_feat_transform.cpu() < random_neg) * (-clip)
                this_feat_transform = this_feat_transform.float()
                if args.device == 'cuda':
                    this_feat_transform = this_feat_transform.cuda()
                feat_transform_grad[layer] += this_feat_transform
              # obtain layer aggregation gradients
              this_layer_grad = this_model.layer_agg_weight.grad
              this_layer_grad = torch.clamp(this_layer_grad, -0.5, 0.5)
              random_neg = np.random.uniform(low=-args.clip, high=0, size=this_layer_grad.shape)
              random_pos = np.random.uniform(low=0, high=args.clip, size=this_layer_grad.shape)
              random_neg = torch.from_numpy(random_neg).float()
              random_pos = torch.from_numpy(random_pos).float()
              this_layer_grad = (this_layer_grad.cpu() > random_pos) * args.clip + (this_layer_grad.cpu() < random_neg) * (-args.clip)
              this_layer_grad = this_layer_grad.float()
              if args.device == 'cuda':
                  this_layer_grad = this_layer_grad.cuda()
              layer_agg_grad += this_layer_grad
              # obtain similarity gradients
              this_a_grad = this_model.a.grad
              this_a_grad = torch.clamp(this_a_grad, -0.5, 0.5)
              random_neg = np.random.uniform(low=-args.clip, high=0, size=this_a_grad.shape)
              random_pos = np.random.uniform(low=0, high=args.clip, size=this_a_grad.shape)
              random_neg = torch.from_numpy(random_neg).float()
              random_pos = torch.from_numpy(random_pos).float()
              this_a_grad = (this_a_grad.cpu() > random_pos) * args.clip + (this_a_grad.cpu() < random_neg) * (-args.clip)
              this_a_grad = this_a_grad.float()
              if args.device == 'cuda':
                  this_a_grad = this_a_grad.cuda()
              a_grad += this_a_grad
            user_emb_grad = user_emb_grad * cor_ratio
            user_reg_grad = 2 * this_model.embedding_user.weight.data/dataset.n_users
            user_emb_grad += user_reg_grad
            # update parameters
            for i in range(args.n_parties):
              this_model = models[i]
              this_model.embedding_user.weight.grad = user_emb_grad
              this_model.layer_agg_weight.grad = layer_agg_grad * cor_ratio
              this_model.a.grad = a_grad  * cor_ratio
              for layer in range(args.n_layers):
                this_model.feat_transform[layer].grad = feat_transform_grad[layer] * cor_ratio
              this_optimizer = optimizers[i]
              this_optimizer.step()

          with torch.no_grad():
            train_result = metrics(models, train_loader)
            test_result = metrics(models, test_loader)
            if (epoch+1) % 10 == 0:
                print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
                          "%H: %M: %S", time.gmtime(time.time()-start_time)))
                print("Train_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(
                          train_result, test_result))

            if test_result < best_rmse:
              best_rmse, best_epoch = test_result, epoch

        print("End. Best epoch {:03d}: Test_RMSE is {:.3f}".format(best_epoch, best_rmse))

def train_GCN(train_loader, test_loader, models, itemID_parties, optimizers, dataset, phi, args):
        count, best_rmse = 0, 100
        for epoch in range(args.epochs):
          start_time = time.time()
          center_party = np.random.choice(args.n_parties)
          for users, items, ratings in train_loader:
            selected_parties = np.random.choice(args.n_parties, args.n_ptcp, replace=False)

            # compute gradients
            user_emb_grad = 0
            reduced_user_emb_grad = 0
            layer_agg_grad = 0
            feat_transform_grad = {i:0 for i in range(args.n_layers)}
            useremb_aggs = torch.zeros(args.n_parties, args.n_layers, dataset.n_users, args.latent_dim)

            for layer in range(args.n_layers):
              for i in selected_parties:
                this_model = models[i]
                useremb_aggs[i, layer] = this_model.users_emb_agg[layer].data

            num_item_choose = 0
            for i in selected_parties:
                num_item_choose += len(itemID_parties[i])
            cor_ratio = dataset.m_items/num_item_choose

            for i in selected_parties:
                last_item_idx = 0
                for j in range(i):
                    last_item_idx += len(itemID_parties[j])
                all_items = itemID_parties[i]
                this_idx = torch.isin(items, all_items)
                this_idx = this_idx.nonzero(as_tuple=True)[0]
                this_users = users[this_idx]
                this_items = items[this_idx] - last_item_idx
                this_ratings = ratings[this_idx]
                this_model = models[i]
                this_model.train()
                this_model.zero_grad()
                this_useremb_aggs = 0
                for j in selected_parties:
                  if j!=i:
                    this_useremb_aggs += useremb_aggs[j]
                this_useremb_aggs = torch.einsum('pu,huk->hpk', phi, this_useremb_aggs) * cor_ratio
                prediction = this_model(this_users, this_items, this_useremb_aggs)
                loss = this_model.sqr_loss(this_ratings, prediction)
                loss.backward()
                # obtain user embedding gradients
                this_user_emb_grad = this_model.embedding_user.weight.grad
                this_user_emb_grad = torch.clamp(this_user_emb_grad, -0.5, 0.5)
                random_neg = np.random.uniform(low=-args.clip, high=0, size=this_user_emb_grad.shape)
                random_pos = np.random.uniform(low=0, high=args.clip, size=this_user_emb_grad.shape)
                random_neg = torch.from_numpy(random_neg).float()
                random_pos = torch.from_numpy(random_pos).float()
                this_user_emb_grad = (this_user_emb_grad > random_pos) * args.clip + (this_user_emb_grad < random_neg) * (-args.clip)
                this_user_emb_grad = this_user_emb_grad.float()
                user_emb_grad += this_user_emb_grad
              # obtain feature transform gradients
                for layer in range(args.n_layers):
                    this_feat_transform = this_model.feat_transform[layer].weight.grad
                    this_feat_transform = torch.clamp(this_feat_transform, -0.5, 0.5)
                    random_neg = np.random.uniform(low=-args.clip, high=0, size=this_feat_transform.shape)
                    random_pos = np.random.uniform(low=0, high=args.clip, size=this_feat_transform.shape)
                    random_neg = torch.from_numpy(random_neg).float()
                    random_pos = torch.from_numpy(random_pos).float()
                    this_feat_transform = (this_feat_transform > random_pos) * args.clip + (this_feat_transform < random_neg) * (-args.clip)
                    this_feat_transform = this_feat_transform.float()
                    feat_transform_grad[layer] += this_feat_transform
              # obtain layer aggregation gradients
                this_layer_grad = this_model.layer_agg_weight.grad
                this_layer_grad = torch.clamp(this_layer_grad, -0.5, 0.5)
                random_neg = np.random.uniform(low=-args.clip, high=0, size=this_layer_grad.shape)
                random_pos = np.random.uniform(low=0, high=args.clip, size=this_layer_grad.shape)
                random_neg = torch.from_numpy(random_neg).float()
                random_pos = torch.from_numpy(random_pos).float()
                this_layer_grad = (this_layer_grad > random_pos) * args.clip + (this_layer_grad < random_neg) * (-args.clip)
                this_layer_grad = this_layer_grad.float()
                layer_agg_grad += this_layer_grad
            user_emb_grad =  user_emb_grad * cor_ratio
            user_reg_grad = 2 * this_model.embedding_user.weight.data/dataset.n_users
            user_emb_grad += user_reg_grad
            # update parameters
            for i in range(args.n_parties):
              this_model = models[i]
              this_model.embedding_user.weight.grad = user_emb_grad
              this_model.layer_agg_weight.grad = layer_agg_grad * cor_ratio
              for layer in range(args.n_layers):
                this_model.feat_transform[layer].weight.grad = feat_transform_grad[layer] * cor_ratio
              this_optimizer = optimizers[i]
              this_optimizer.step()
          with torch.no_grad():
              train_result = metrics(models, train_loader)
              test_result = metrics(models, test_loader)
              if test_result < best_rmse:
                best_rmse, best_epoch = test_result, epoch
        print("End. Best epoch {:03d}: Test_RMSE is {:.3f}".format(best_epoch, best_rmse))

def train_GGNN(train_loader, test_loader, models, itemID_parties, optimizers, dataset, phi, args):
    count, best_rmse = 0, 100
    for epoch in range(args.epochs):
      start_time = time.time()
      for users, items, ratings in train_loader:
        selected_parties = np.random.choice(args.n_parties, args.n_ptcp, replace=False)
        # compute gradients
        user_emb_grad = 0
        layer_agg_grad = 0
        W_grad = 0
        Wr_grad = 0
        Wz_grad = 0
        U_grad = 0
        Ur_grad = 0
        Uz_grad = 0
        useremb_aggs = torch.zeros(args.n_parties, args.n_layers, dataset.n_users, args.latent_dim)
        for layer in range(args.n_layers):
          for i in selected_parties:
            this_model = models[i]
            useremb_aggs[i, layer] = this_model.users_emb_agg[layer].data
            
        num_item_choose = 0
        for i in selected_parties:
            num_item_choose += len(itemID_parties[i])
        cor_ratio = dataset.m_items/num_item_choose
            
        for i in selected_parties:
          last_item_idx = 0
          for j in range(i):
            last_item_idx += len(itemID_parties[j])
          all_items = itemID_parties[i]
          this_idx = torch.isin(items, all_items)
          this_idx = this_idx.nonzero(as_tuple=True)[0]
          this_users = users[this_idx]
          this_items = items[this_idx] - last_item_idx
          this_ratings = ratings[this_idx]
          if args.device == 'cuda':
            this_users = this_users.cuda()
            this_items = this_items.cuda()
            this_ratings = this_ratings.cuda()
          this_model = models[i]
          this_model.train()
          this_model.zero_grad()
          this_useremb_aggs = 0
          for j in selected_parties:
            if j!=i:
              this_useremb_aggs += useremb_aggs[j]
          if args.device == 'cuda':
            this_useremb_aggs = this_useremb_aggs.cuda()
          this_useremb_aggs = torch.einsum('pu,huk->hpk', phi, this_useremb_aggs)
          prediction = this_model(this_users, this_items, this_useremb_aggs)
          loss = this_model.sqr_loss(this_ratings, prediction)
          loss.backward()
          # obtain user embedding gradients
          this_user_emb_grad = this_model.embedding_user.weight.grad
          this_user_emb_grad = torch.clamp(this_user_emb_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_user_emb_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_user_emb_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_user_emb_grad = (this_user_emb_grad.cpu() > random_pos) * args.clip + (this_user_emb_grad.cpu() < random_neg) * (-args.clip)
          this_user_emb_grad = this_user_emb_grad.float()
          if args.device == 'cuda':
            this_user_emb_grad = this_user_emb_grad.cuda()
          user_emb_grad += this_user_emb_grad
          # obtain feature transform gradients
          this_W_grad = this_model.W.grad
          this_W_grad = torch.clamp(this_W_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_W_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_W_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_W_grad = (this_W_grad.cpu() >= random_pos) * args.clip + (this_W_grad.cpu() <= random_neg) * (-args.clip)
          this_Wr_grad = this_model.Wr.grad
          this_Wr_grad = torch.clamp(this_Wr_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_Wr_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_Wr_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_Wr_grad = (this_Wr_grad.cpu() >= random_pos) * args.clip + (this_Wr_grad.cpu() <= random_neg) * (-args.clip)
          this_Wz_grad = this_model.Wz.grad
          this_Wz_grad = torch.clamp(this_Wz_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_Wz_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_Wz_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_Wz_grad = (this_Wz_grad.cpu() >= random_pos) * args.clip + (this_Wz_grad.cpu() <= random_neg) * (-args.clip)
          this_U_grad = this_model.U.grad
          this_U_grad = torch.clamp(this_U_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_U_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_U_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_U_grad = (this_U_grad.cpu() >= random_pos) * args.clip + (this_U_grad.cpu() <= random_neg) * (-args.clip)
          this_Ur_grad = this_model.Ur.grad
          this_Ur_grad = torch.clamp(this_Ur_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_Ur_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_Ur_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_Ur_grad = (this_Ur_grad.cpu() >= random_pos) * args.clip + (this_Ur_grad.cpu() <= random_neg) * (-args.clip)
          this_Uz_grad = this_model.Uz.grad
          this_Uz_grad = torch.clamp(this_Uz_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_Uz_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_Uz_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_Uz_grad = (this_Uz_grad.cpu() >= random_pos) * args.clip + (this_Uz_grad.cpu() <= random_neg) * (-args.clip)
          if args.device == 'cuda':
            this_W_grad = this_W_grad.cuda()
            this_Wr_grad = this_Wr_grad.cuda()
            this_Wz_grad = this_Wz_grad.cuda()
            this_U_grad = this_U_grad.cuda()
            this_Ur_grad = this_Ur_grad.cuda()
            this_Uz_grad = this_Uz_grad.cuda()
          W_grad += this_W_grad.float()
          Wr_grad += this_Wr_grad.float()
          Wz_grad += this_Wz_grad.float()
          U_grad += this_U_grad.float()
          Ur_grad += this_Ur_grad.float()
          Uz_grad += this_Uz_grad.float()
          # obtain layer aggregation gradients
          this_layer_grad = this_model.layer_agg_weight.grad
          this_layer_grad = torch.clamp(this_layer_grad, -0.5, 0.5)
          random_neg = np.random.uniform(low=-args.clip, high=0, size=this_layer_grad.shape)
          random_pos = np.random.uniform(low=0, high=args.clip, size=this_layer_grad.shape)
          random_neg = torch.from_numpy(random_neg).float()
          random_pos = torch.from_numpy(random_pos).float()
          this_layer_grad = (this_layer_grad.cpu() > random_pos) * args.clip + (this_layer_grad.cpu() < random_neg) * (-args.clip)
          this_layer_grad = this_layer_grad.float()
          if args.device == 'cuda':
              this_layer_grad = this_layer_grad.cuda()
          layer_agg_grad += this_layer_grad
        user_emb_grad = user_emb_grad * cor_ratio
        user_reg_grad = 2 * this_model.embedding_user.weight.data/dataset.n_users
        user_emb_grad += user_reg_grad
        # update parameters
        for i in range(args.n_parties):
          this_model = models[i]
          this_model.embedding_user.weight.grad = user_emb_grad
          this_model.layer_agg_weight.grad = layer_agg_grad * cor_ratio
          this_model.W.grad = W_grad * cor_ratio
          this_model.Wr.grad = Wr_grad * cor_ratio
          this_model.Wz.grad = Wz_grad * cor_ratio
          this_model.U.grad = U_grad * cor_ratio
          this_model.Ur.grad = Ur_grad * cor_ratio
          this_model.Uz.grad = Uz_grad * cor_ratio
          this_optimizer = optimizers[i]
          this_optimizer.step()
      with torch.no_grad():
        train_result = metrics(models, train_loader)
        test_result = metrics(models, test_loader)
        if (epoch+1) % 10 == 0:
          print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
                    "%H: %M: %S", time.gmtime(time.time()-start_time)))
          print("Train_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(
                    train_result, test_result))

        if test_result < best_rmse:
          best_rmse, best_epoch = test_result, epoch