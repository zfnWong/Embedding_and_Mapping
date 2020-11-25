"""

"""
import torch
from torch import nn
import torch.nn.functional as F
import copy
from parse import args


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, graph, users):
        raise NotImplementedError

class PureMF(BasicModel):
    def __init__(self, dataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = args.latent_dim_rec
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, graph, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class NGCF(BasicModel):
    def __init__(self, dataset):
        super(NGCF, self).__init__()
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.embed_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.sizes = [self.embed_dim] + args.layer_size
        self.keep_prob = 1 - args.drop_rate
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)
        self.conv = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(self.n_layers)])
        self.ew = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(self.n_layers)])

        self.__init_weight()

    def __init_weight(self):
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        for conv, ew in zip(self.conv, self.ew):
            nn.init.xavier_uniform_(conv.weight)
            nn.init.xavier_uniform_(ew.weight)

    def __dropout(self, graph, keep_prob):
        """
        use dropout to make a more sparse graph
        """
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        if args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(graph, self.keep_prob)
            else:
                g_droped = graph
        else:
            g_droped = graph

        embs = [all_emb]
        for layer in range(self.n_layers):
            side_embeddings = torch.sparse.mm(g_droped, all_emb)
            out_conv = self.conv[layer](side_embeddings)
            out_ew = self.ew[layer](torch.mul(side_embeddings, all_emb))
            all_emb = F.leaky_relu(out_conv + out_ew, negative_slope=0.2)
            """
            notice!!!
            normalize before every layer's output
            """
            norm_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(norm_emb)

        embs = torch.cat(embs, dim=1)  # (n_nodes, embed_size * n_layers)

        users, items = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return users, items

    def getUsersRating(self, graph, users):
        all_users, all_items = self.computer(graph)
        all_users = all_users[users.long()]  # (n_nodes, embed_size)
        # why sigmoid here? it meet the need of sklearn metric
        rating = torch.sigmoid(torch.matmul(all_users, all_items.t()))
        return rating  # (n_users, n_items)

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos_items)
        negEmb0 = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        return


class LightGCN(BasicModel):
    def __init__(self, dataset):
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.drop_rate
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.__init_weight()

    def __init_weight(self):
        """
        random normal init seems to be a better choice
        when lightGCN actually don't use any non-linear activation function
        """
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # new characteristic of python 3.6 - f""
        print(f"model lgn(dropout:{args.dropout})")

    def __dropout(self, graph, keep_prob):
        """
        use dropout to make a more sparse graph
        """
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        if args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(graph, self.keep_prob)
            else:
                g_droped = graph
        else:
            g_droped = graph

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=2)  # (n_nodes, embed_size, n_layers)
        embs = torch.mean(embs, dim=2)

        users, items = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return users, items

    def getUsersRating(self, graph, users):
        all_users, all_items = self.computer(graph)
        all_users = all_users[users.long()]  # (n_nodes, embed_size)
        # why sigmoid here?
        rating = torch.sigmoid(torch.matmul(all_users, all_items.t()))
        return rating  # (n_users, n_items)

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # why softplus here?
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        return


class LightGCN_ws(BasicModel):
    def __init__(self, dataset):
        super(LightGCN_ws, self).__init__()
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.drop_rate
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # weighted sum
        self.user_weight = nn.Linear(self.n_layers+1, 1, bias=False)
        self.item_weight = nn.Linear(self.n_layers+1, 1, bias=False)

        self.__init_weight()

    def __init_weight(self):
        """
        random normal init seems to be a better choice
        when lightGCN actually don't use any non-linear activation function
        """
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.user_weight.weight, std=0.1)
        nn.init.normal_(self.item_weight.weight, std=0.1)

        # new characteristic of python 3.6 - f""
        print(f"model lgn(dropout:{args.dropout})")

    def __dropout(self, graph, keep_prob):
        """
        use dropout to make a more sparse graph
        """
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        if args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(graph, self.keep_prob)
            else:
                g_droped = graph
        else:
            g_droped = graph

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=2)  # (n_nodes, embed_size, n_layers)

        users, items = torch.split(embs, [self.num_users, self.num_items], dim=0)
        users = self.user_weight(users).squeeze()
        items = self.item_weight(items).squeeze()

        return users, items

    def getUsersRating(self, graph, users):
        all_users, all_items = self.computer(graph)
        all_users = all_users[users.long()]  # (n_nodes, embed_size)
        # why sigmoid here?
        rating = torch.sigmoid(torch.matmul(all_users, all_items.t()))
        return rating  # (n_users, n_items)

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # why softplus here?
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        return


class LightGCN_ecc(BasicModel):
    def __init__(self, dataset):
        super(LightGCN_ecc, self).__init__()
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.drop_rate
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # bn
        tmp = nn.BatchNorm1d(num_features=self.latent_dim, momentum=0.1)
        self.u_b = clones(tmp, self.n_layers + 1)
        self.i_b = clones(tmp, self.n_layers + 1)

        # ecc 是否用softmax  是否先batchnorm  是否先dropout  是否sigmoid tanh
        self.ecc_layer = args.ecc_layer
        assert self.ecc_layer > 0
        hidden = self.latent_dim
        dropout_rate = 0.5
        if args.ecc:
            current_dim = (self.n_layers + 1) ** 2 * self.latent_dim
        else:
            current_dim = (self.n_layers + 1) ** 2
        if self.ecc_layer == 1:
            last_dim = current_dim
        else:
            last_dim = hidden
        ecc_list = []
        for i in range(self.ecc_layer):
            if i == self.ecc_layer - 1:
                ecc_list.append(nn.Linear(in_features=last_dim, out_features=1))
            else:
                ecc_list.extend([
                    nn.Linear(in_features=current_dim, out_features=hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = hidden
        self.ecc = nn.Sequential(*ecc_list)
        self.__init_weight()

    def __init_weight(self):
        """
        random normal init seems to be a better choice
        when lightGCN actually don't use any non-linear activation function
        """
        if self.ecc_layer > 1:
            nn.init.xavier_uniform_(self.embedding_user.weight)
            nn.init.xavier_uniform_(self.embedding_item.weight)
        else:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.ecc_layer > 1:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.normal_(m.weight, std=0.1)

        # new characteristic of python 3.6 - f""
        print(f"model lgn(dropout:{args.dropout})")

    def __dropout(self, graph, keep_prob):
        """
        use dropout to make a more sparse graph
        """
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(graph, self.keep_prob)
            else:
                g_droped = graph
        else:
            g_droped = graph

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # (n_nodes, n_layers, embed_size)

        users, items = torch.split(embs, [self.num_users, self.num_items])
        # whether use bn to rep. of every layer
        # for i in range(self.n_layers + 1):
        #     users[:, i, :] = self.u_b[i](users[:, i, :])
        #     items[:, i, :] = self.i_b[i](items[:, i, :])
        return users, items

    def getUsersRating(self, graph, users):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users.long()]  # (n_nodes, n_layers, embed_size)
        items_emb = all_items.permute(1, 0, 2)  # (n_layers, n_nodes, embed_size)

        if args.ecc:
            layer_cross_mul = [torch.mul(layer1, layer2).cpu() for u in users_emb for layer1 in u for layer2 in items_emb]
        else:
            layer_cross_mul = [torch.mul(layer1, layer2).sum(dim=1, keepdim=True).cpu() for u in users_emb for layer1 in u for layer2 in items_emb]
        layer_cross_mul = torch.stack(layer_cross_mul, dim=0)
        layer_cross_mul = layer_cross_mul.view(len(users), -1, self.num_items, 1).transpose(0, 1)
        # (n_users, n_items, embed_len * n_pairs)
        layer_cross_mul = torch.cat([c for c in layer_cross_mul], dim=-1).view(len(users) * self.num_items, -1)

        rating = []
        n_fold = args.n_fold
        fold_len = len(layer_cross_mul) // n_fold
        for i in range(n_fold):
            if i == n_fold - 1:
                rating.append(torch.sigmoid(self.ecc(layer_cross_mul[i * fold_len:].to(args.device))))
            else:
                rating.append(
                    torch.sigmoid(self.ecc(layer_cross_mul[i * fold_len: (i + 1) * fold_len].to(args.device))))
        rating = torch.cat(rating, dim=0).view(len(users), self.num_items)

        return rating

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = \
            self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = \
            (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        users_emb, pos_emb, neg_emb = users_emb.permute(1, 0, 2), pos_emb.permute(1, 0, 2), neg_emb.permute(1, 0, 2)

        if args.ecc:
            pos_scores = [torch.mul(layer1, layer2) for layer1 in users_emb for layer2 in pos_emb]
            neg_scores = [torch.mul(layer1, layer2) for layer1 in users_emb for layer2 in neg_emb]
        else:
            pos_scores = [torch.mul(layer1, layer2).sum(1, keepdim=True) for layer1 in users_emb for layer2 in pos_emb]
            neg_scores = [torch.mul(layer1, layer2).sum(1, keepdim=True) for layer1 in users_emb for layer2 in neg_emb]

        pos_scores = self.ecc(torch.cat(pos_scores, dim=1)).squeeze()
        neg_scores = self.ecc(torch.cat(neg_scores, dim=1)).squeeze()
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        return