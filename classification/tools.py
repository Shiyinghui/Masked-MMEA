from collections import defaultdict
from PIL import Image
from config import Config
import torch
import numpy as np


def get_ent2id(file_path):
    ent2id = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ent2id[th[1]] = int(th[0])
    return ent2id


def get_ills(file_path):
    ills = []
    #ills = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            id1, id2 = line.strip().split('\t')
            ills.append((int(id1), int(id2)))
            #ills[int(id1)] = int(id2)
    return ills


def read_triples(file_path):
    tuples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            params = line.strip("\n").split("\t")
            tuples.append(tuple([int(x) for x in params]))
    return tuples


def get_ent2type(files: list):
    """
    files: a list of file paths
    """
    ent2type = dict()
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                data = line.strip().split('\t')
                if len(data) > 1:
                    ent, type = data[0:2]
                    ent2type[ent] = type.split('/')[-1]
                    #ent2type[ent] = type
    return ent2type


def get_ent_type2cnt(ent_set: set, ent2type: dict):
    ent_type2cnt = dict()
    for ent in ent_set:
        if ent in ent2type:
            if ent2type[ent] not in ent_type2cnt:
                ent_type2cnt[ent2type[ent]] = 0
            ent_type2cnt[ent2type[ent]] += 1
    return ent_type2cnt


# key: type, value: a list of entities of this type
def get_type2list(ent2type):
    type2list = defaultdict(list)
    for k, v in ent2type.items():
        type2list[v].append(k)

    return type2list


def get_ent_imgs(file_path):
    ent_with_img = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ent_with_img.add(line.strip())
    return ent_with_img


def get_ill_pair(kg1_ent2id_path, kg2_ent2id_path, ill_path):
    kg1_ent2id = get_ent2id(kg1_ent2id_path)
    kg2_ent2id = get_ent2id(kg2_ent2id_path)
    ill = get_ills(ill_path)

    ent2id = dict(kg1_ent2id, **kg2_ent2id)
    id2ent = {v: k for k, v in ent2id.items()}
    ill_ent_pair = [(id2ent[id1], id2ent[id2]) for (id1, id2) in ill]
    return ill_ent_pair


def get_ent2degree(triple_path):
    triples = read_triples(triple_path)
    rel2cnt = defaultdict(int)
    ent2degree = defaultdict(int)
    for tri in triples:
        ent1, rel, ent2 = tri[:]
        rel2cnt[rel] += 1
        ent2degree[ent1] += 1
        ent2degree[ent2] += 1
    print('avg degree: ', sum(ent2degree.values()) / len(ent2degree))
    return ent2degree


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def get_ent_id(cfg: Config):
    kg1_ent2id: dict = get_ent2id(cfg.kg1_ent2id_path)
    kg2_ent2id: dict = get_ent2id(cfg.kg2_ent2id_path)
    ent2id = dict(kg1_ent2id, **kg2_ent2id)
    id2ent = {v: k for k, v in ent2id.items()}
    return ent2id, id2ent


def get_img_data(pil_image):
    data_shape = np.array(pil_image).shape
    if len(data_shape) == 2:  # gray image
        return pil_image.convert('RGB')
    else:  # len(data_shape)=3
        if data_shape[-1] == 4:  # four channel, image mode: RGBA
            return pil_image.convert('RGB')
        elif data_shape[-1] == 2:  # two channel, image mode: LA
            tmp = np.expand_dims(np.array(pil_image)[:, :, 0], axis=2)
            return Image.fromarray(np.concatenate([tmp, tmp, tmp], axis=2))
        else:  # valid, three channel, RGB
            return pil_image


