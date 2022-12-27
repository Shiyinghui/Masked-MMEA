from process_type import get_type2path, get_sub2super
from collections import defaultdict
from config import Config
from tools import *
import os
import pickle


def get_ft_info(cfg):
    # read type predictions
    data_path = cfg.ill_img_data_path
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    type_set = set()
    type2idx = dict()
    # item: [ent, type, img_name, label_id]
    for item in data:
        type2idx[item[1]] = item[3]
        type_set.add(item[1])
    return type_set, type2idx


def get_disjoint_type(classes, data_path):
    """
    :param classes: set of types
    :param data_path: disjoint relations defined by owl
    """
    sub2super = get_sub2super()
    type2path: dict = get_type2path(sub2super)
    disjoint = defaultdict(set)

    with open(data_path, 'r') as f:
        for line in f.readlines():
            h, r, t = line.strip('\n').split('\t')
            h = h.split('/')[-1]
            t = t.split('/')[-1]
            if h in classes and t in classes:
                disjoint[h].add(t)
                disjoint[t].add(h)

    # set direct subclasses of owl:Thing as mutual disjoint
    l1_class = list({path[-2] for path in type2path.values() if len(path) >= 2})
    for i in range(len(l1_class)-1):
        for j in range(i+1, len(l1_class)):
            disjoint[l1_class[i]].add(l1_class[j])
            disjoint[l1_class[j]].add(l1_class[i])

    # set Person and Organisation as disjoint
    disjoint['Person'].add('Organisation')
    disjoint['Organisation'].add('Person')

    # expand disjoint relation ship
    for t in classes:
        h_types = type2path[t][1:-1]
        for h_t in h_types:
            if h_t in disjoint:
                disjoint[t] = disjoint[t].union(disjoint[h_t])
                for tail in disjoint[t]:
                    disjoint[tail].add(t)
    return disjoint


def get_raw_ent_type(cfg):
    type2entset = defaultdict(set)
    type_data_dir = cfg.ent_type_dir
    for split in ['fr_en', 'ja_en', 'zh_en']:
        for fn in os.listdir(os.path.join(type_data_dir, split)):
            if 'link' not in fn:
                continue
            with open(os.path.join(type_data_dir, split, fn), 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in data:
                ent, t = line.strip().split('\t')
                type2entset[t.split('/')[-1]].add(ent.split('/')[-1])
    return type2entset


def build_conflict_matrix(cfg):
    type_set, type2idx = get_ft_info(cfg)
    type2idx = sorted(type2idx.items(), key=lambda item: item[1])
    idx2type = {v: k for k, v in type2idx}
    label_ind = list(idx2type.values())
    conflict = dict()

    sub2super = get_sub2super()
    type2path: dict = get_type2path(sub2super)
    disjoint = get_disjoint_type(type_set, cfg.disjoint_info)
    type2entset = get_raw_ent_type(cfg)

    for i in range(len(label_ind)):
        conflict[(i, i)] = 0
    for i in range(len(label_ind)-1):
        for j in range(i+1, len(label_ind)):
            t_i, t_j = idx2type[i], idx2type[j]
            path_i, path_j = set(type2path[t_i]), set(type2path[t_j])
            # condition 1, check if type t_i and type t_j are disjoint
            if t_i in disjoint and t_j in disjoint[t_i]:
                conflict[(i, j)] = 1

            # condition 2, consider types in a line path not disjoint
            elif t_i in path_j or t_j in path_i:
                conflict[(i, j)] = 0

            # condition 3, if type i and type j have at least one common entity,
            # consider them not disjoint
            elif len(type2entset[t_i].intersection(type2entset[t_j])):
                conflict[(i, j)] = 0

            # condition 4, calculate conflict degree
            else:
                path_i, path_j = set(type2path[t_i]), set(type2path[t_j])
                conflict[(i, j)] = 1 - len(path_i.intersection(path_j))/len(path_i.union(path_j))
            conflict[(j, i)] = conflict[(i, j)]

    conflict_val = set(conflict.values())
    # for i in conflict_val:
    #     print(i)
    return conflict


def generate_mask(cfg, conflict_thresh=0.0, strict=False, save_mask=False):
    """
    :param cfg: config of file paths
    :param conflict_thresh: conflict threshold
    :param strict: set to false, if top five predicted labels contains the true label of an image,
    consider this prediction correct
    :param save_mask: if True, save this mask
    """
    val_data_path = cfg.ill_img_data_path
    pred_data_path = cfg.pred_path

    mask_path = cfg.mask_path
    if conflict_thresh > 0:
        mask_path = cfg.mask_path.split('.')[0] + '_' + str(conflict_thresh) + '.pkl'

    conflict = build_conflict_matrix(cfg)
    with open(val_data_path, 'rb') as f:
        val_data: list = pickle.load(f)  # [ent, label, img_name, label_idx]
    entities = set([item[0] for item in val_data])
    with open(pred_data_path, 'rb') as f:
        pred_data: dict = pickle.load(f)  # img_name: label_idx
    gt_info = {item[2]: [item[3], item[0], item[1]] for item in val_data}
    mask_ent = {ent: 0 for ent in entities}

    for k, v in pred_data.items():
        gt_label_ind = gt_info[k][0]
        gt_ent = gt_info[k][1]
        min_conflict = conflict[(v[0], gt_label_ind)]
        if not strict:
            min_conflict = min([conflict[(i, gt_label_ind)] for i in v])
        if min_conflict <= conflict_thresh:
            mask_ent[gt_ent] = 1

    print(sum(list(mask_ent.values())))
    if save_mask:
        with open(mask_path, 'wb') as f:
            pickle.dump(mask_ent, f)
    return mask_ent


if __name__ == "__main__":
    split = 'fr_en'  # 'fr_en' or 'ja_en' or 'zh_en'
    cfg = Config(split)
    build_conflict_matrix(cfg)
    generate_mask(cfg, conflict_thresh=0.4, strict=False, save_mask=False)