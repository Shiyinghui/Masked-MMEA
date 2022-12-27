from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import torch
import pickle


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def read_raw_data(file_dir, l=[1, 2]):
    print('loading raw data...')
    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups
    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in l])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in l])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
    csls_sim_mat = 2 * sim_mat.t() - nearest_values1
    csls_sim_mat = csls_sim_mat.t() - nearest_values2
    return csls_sim_mat


def get_topk_indices(M, K=1000):
    H, W = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    print ("highest sim:", vals[0].item(), "lowest sim:", vals[-1].item())
    two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return two_d_indices


def normalize_zero_one(A):
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0]
    return A


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    #img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    #img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])
    #img_embd = np.array([img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    #print("%.2f%% entities have images" % (100 * len(img_dict)/e_num))
    #return img_embd
    img_vec = []
    mask = torch.ones(e_num, dtype=torch.float32)
    for i in range(e_num):
        if i in img_dict:
            img_vec.append(img_dict[i])
        else:
            img_vec.append(np.random.normal(mean, std, mean.shape[0]))
            mask[i] = 0
    img_vec = np.array(img_vec)
    print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
    return img_vec, mask


if __name__ == '__main__':
    pass

