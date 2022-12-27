from tools import *
from config import Config
import re


def get_sub2super():
    cfg = Config()
    onto_file = cfg.ontology_file_path
    sub2sup_class = dict()
    ptn = re.compile('<(.*?)>')
    with open(onto_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            sub_c, rel, sup_c = line.strip().split()[:3]
            sub_c = ptn.findall(sub_c)[0].split('/')[-1]
            sup_c = ptn.findall(sup_c)[0].split('/')[-1]
            sub2sup_class[sub_c] = sup_c
    return sub2sup_class


def get_type2desc():
    sub2sup_class: dict = get_sub2super()
    super2sub = defaultdict(set)
    type2des = defaultdict(set)
    for k, v in sub2sup_class.items():
        super2sub[v].add(k)
    for k, v in super2sub.items():
        s = list(v)
        while len(s) > 0:
            elem = s.pop(0)
            type2des[k].add(elem)
            if elem in super2sub:
                s.extend(super2sub[elem])
    return type2des

# if __name__ == "__main__":
#     get_type2desc()


def get_sibling():
    sub2sup_class: dict = get_sub2super()
    super2sub = defaultdict(set)
    siblings = defaultdict(set)
    for k, v in sub2sup_class.items():
        super2sub[v].add(k)
    for k, v in super2sub.items():
        for elem in v:
            siblings[elem] = v - {elem}
    return siblings


def get_type2path(sub2super: dict):
    """
    :param sub2super: key: an entity type, value: its super class
    """
    type2path = dict()
    for k in sub2super.keys():
        leaf = k
        hierarchical_class = [k]
        while True:
            if k in sub2super:
                hierarchical_class.append(sub2super[k])
                k = sub2super[k]
            else:
                break
        type2path[leaf] = hierarchical_class
    return type2path


def reclassify(ent2type, sub2super: dict, thresh=30, depth=4):
    """
    if the number of entities of one type is below a threshold, reclassify these entities
    :param ent2type: dict, storing entities and their types(classes)
    :param sub2super: dict, storing classes and their super classes
    :param thresh: minimum number of entities for a type
    :param depth: starting depth
    """
    type2list = get_type2list(ent2type)
    type2path: dict = get_type2path(sub2super)
    type2level = {k: len(v) for k, v in type2path.items()}
    h = depth  # 4
    while h > 2:
        temp_type = {k for k, v in type2list.items() if len(v) < thresh}
        for k in temp_type:
            if type2level[k] == h:
                if sub2super[k] in type2list:
                    type2list[sub2super[k]].extend(type2list[k])
                else:
                    type2list[sub2super[k]] = type2list[k]
                type2list.pop(k)
        h -= 1
    return type2list


def raise_type(ent2type, type2path, depth=4):
    """
    :param ent2type: dict, key: ent, value: its type
    :param type2path: dict, key: type, value: a list of types
    :param depth: the depth of a type in a hierarchical type tree
    """
    for ent, t in ent2type.items():
        res_type = type2path[t][-depth] if len(type2path[t]) >= depth else t
        ent2type[ent] = res_type
    return ent2type


def assign_type(ill_ent_pair: list, ent2type: dict):
    """
    for two aligned entities a and b, let the type of a (if not given) be the type of b
    :param ill_ent_pair: a list of aligned entity pairs
    :param ent2type:
    :return:
    """
    for (ent1, ent2) in ill_ent_pair:
        if ent1 not in ent2type and ent2 in ent2type:
            ent2type[ent1] = ent2type[ent2]
        if ent1 in ent2type and ent2 not in ent2type:
            ent2type[ent2] = ent2type[ent1]
    return ent2type


def handle_diff(ent2type: dict, ill_ent_pair, type2level: dict, sub2super: dict):
    """
    if two aligned entities have different types, make the finest type their final type
    """
    for (ent1, ent2) in ill_ent_pair:
        if ent1 in ent2type and ent2 in ent2type:
            t1 = ent2type[ent1]
            t2 = ent2type[ent2]
            if t1 != t2:
                if type2level[t1] > type2level[t2]:
                    ent2type[ent2] = t1
                if type2level[t1] < type2level[t2]:
                    ent2type[ent1] = t2
                # else:
                #     if sub2super[t1] == sub2super[t2]:
                #         ent2type[ent1] = sub2super[t1]
                #         ent2type[ent2] = sub2super[t1]
    return ent2type




