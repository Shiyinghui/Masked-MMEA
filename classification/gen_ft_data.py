from process_type import *
from tools import get_ent2type
import os
import pickle


def get_all_ent_type():
    # get classes for entities
    fl = [os.path.join(cfg.ent_type_dir, 'fr_en', file)
          for file in ['fr_crossview_link', 'en_crossview_link']]
    ent2type = get_ent2type(fl)
    for sp in ['ja_en', 'zh_en']:
        if sp == 'ja_en':
            fl = [os.path.join(cfg.ent_type_dir, sp, file)
                  for file in ['ja_crossview_link', 'en_crossview_link']]
            tmp_data = get_ent2type(fl)
        else:
            tmp_data = get_ent2type([os.path.join(cfg.ent_type_dir, sp, 'en_crossview_link')])
        for k, v in tmp_data.items():
            if k in ent2type:
                continue
            ent2type[k] = v
    return ent2type


# given a split, get training and testing sets of entities, and aligned pairs
def get_train_test(ent2imgfn: dict, split='fr_en'):
    ill_ent_pair = []
    img_set = set(ent2imgfn.keys())
    ill_ent_set, ent_set = set(), set()
    ill_split2ent = dict()
    for ds_split in ['fr_en', 'ja_en', 'zh_en']:
        cfg = Config(ds_split)
        kg1_ent2id = get_ent2id(cfg.kg1_ent2id_path)
        kg2_ent2id = get_ent2id(cfg.kg2_ent2id_path)
        kg_ent = set(kg1_ent2id.keys()).union(kg2_ent2id.keys())

        tmp_ill_ent_pair = get_ill_pair(cfg.kg1_ent2id_path,
                                        cfg.kg2_ent2id_path,
                                        cfg.ills_path)
        tmp_ill_ent_set = set([pair[0] for pair in tmp_ill_ent_pair]).union(
            set([pair[1] for pair in tmp_ill_ent_pair]))
        ill_split2ent[ds_split] = tmp_ill_ent_set

        # update data
        ill_ent_set = ill_ent_set.union(tmp_ill_ent_set)
        ill_ent_pair.extend(tmp_ill_ent_pair)
        ent_set = ent_set.union(kg_ent)

    ent2type = get_all_ent_type()
    ent2type = {k: ent2type[k] for k in ent_set if k in ent2type}
    ent2type = assign_type(ill_ent_pair, ent2type)

    train_set = {ent for ent in (ent_set - ill_split2ent[split]) if ent in img_set}
    test_set = {ent for ent in ill_split2ent[split] if ent in img_set}
    return train_set, test_set, ill_ent_pair, ent2type


def refine_type(ent2type, ill_ent_pair):
    # handle type differences between aligned entities
    sub2super = get_sub2super()
    type2path: dict = get_type2path(sub2super)
    type2level = {k: len(v) for k, v in type2path.items()}
    type2level['owl#Thing'] = 1
    # for k, v in type2level.items():
    #     if v==8:
    #         print(type2path[k])
    # print(set(type2level.values()))

    ent2type = handle_diff(ent2type, ill_ent_pair, type2level, sub2super)

    # delete entities which have 'owl#Thing' as their types
    ents = list(ent2type.keys())
    for k in ents:
        if ent2type[k] == 'owl#Thing':
            ent2type.pop(k)

    type2list = get_type2list(ent2type)
    type2list = sorted(type2list.items(), key=lambda x: len(x[1]), reverse=True)
    for (t, l) in type2list:
        print(type2path[t], len(l))

    type2path: dict = get_type2path(sub2super)
    ent2type = raise_type(ent2type, type2path)

    type2list = reclassify(ent2type, sub2super)
    ent2type = dict()
    for k, v in type2list.items():
        for ent in v:
            ent2type[ent] = k
    return ent2type


def process(ent2imgfn, split='fr_en'):
    # obtain final train data, ill(test) data and ent2type
    train_set, test_set, ill_ent_pair, ent2type = get_train_test(ent2imgfn, split)
    ent2type = refine_type(ent2type, ill_ent_pair)

    train2type = {k: ent2type[k] for k in train_set if k in ent2type}
    train_type2list = get_type2list(train2type)

    test2type = {k: ent2type[k] for k in test_set if k in ent2type}
    test_type2list = get_type2list(test2type)

    return train_type2list, test_type2list, ent2type


def gen_train_test(ent2imgfn, save_dir, split='fr_en'):
    """
    :param ent2imgfn: the mapping from entities to their image names
    :param save_dir: save data to this directory
    """
    train_type2list, ill_type2list, _ = process(ent2imgfn, split)
    train_ent2type, test_ent2type = dict(), dict()

    for ent_type, ent_list in train_type2list.items():
        tmp_list = [ent for ent in ent_list if ent in ent2imgfn]
        tmp_dict = {ent: ent_type for ent in tmp_list}
        train_ent2type.update(tmp_dict)

    for ent_type, ent_list in ill_type2list.items():
        tmp_list = [ent for ent in ent_list if ent in ent2imgfn]
        tmp_dict = {ent: ent_type for ent in tmp_list}
        test_ent2type.update(tmp_dict)

    # only keep common types
    train_types, test_types = set(train_ent2type.values()), set(test_ent2type.values())
    common_types = train_types.intersection(test_types)

    train_ent2type = {k: v for k, v in train_ent2type.items() if v in common_types}
    test_ent2type = {k: v for k, v in test_ent2type.items() if v in common_types}
    # create mapping from type to id
    type2id = {k: i for i, k in enumerate(common_types)}

    train_ent_info = [(k, v, ent2imgfn[k], type2id[v]) for k, v in train_ent2type.items()]
    test_ent_info = [(k, v, ent2imgfn[k], type2id[v]) for k, v in test_ent2type.items()]

    #print(0)
    print(len(train_ent_info), len(test_ent_info))
    with open(os.path.join(save_dir, f'{split}_train.pkl'), 'wb') as f:
        pickle.dump(train_ent_info, f)
    with open(os.path.join(save_dir, f'{split}_test.pkl'), 'wb') as f:
        pickle.dump(test_ent_info, f)


if __name__ == "__main__":
    split = 'zh_en'
    cfg = Config(split)

    ent2img_file = cfg.ent2imgfn_path
    save_ind_dir = cfg.vis_dir
    if not os.path.exists(save_ind_dir):
        os.makedirs(save_ind_dir)
    if os.path.exists(ent2img_file):
        with open(ent2img_file, 'rb') as f:
            ent2fn: dict = pickle.load(f)
    gen_train_test(ent2fn, save_ind_dir, split)
    #process(ent2fn)
    #get_all_ent_type()