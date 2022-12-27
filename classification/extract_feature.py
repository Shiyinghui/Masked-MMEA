from __future__ import print_function
from __future__ import division
import pickle
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
from config import Config
from tools import *


def extract_feature_resnet152(cfg, save_feat=False, save_pred=False):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # load data
    img_data_dir = cfg.img_data_dir
    val_ind_path = cfg.ill_img_data_path
    model_path = cfg.model_path
    save_feat_path = cfg.img_feature_path
    pred_path = cfg.pred_path
    pred_label = dict()

    with open(val_ind_path, 'rb') as f:
        data = pickle.load(f)
        # item: [ent, type, img_name, type_idx]
    images = [os.path.join(img_data_dir, item[2]+'.npy') for item in data]
    fn2ent = {item[2]: item[0] for item in data}
    img2label = {item[2]: item[3] for item in data}

    img_feature_dict = {}
    num_classes = cfg.finetune_classes
    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(model_path, map_location='cpu') #map_location={'cuda:0': 'cuda:1'})
    model_ft.load_state_dict(state_dict)

    layer = model_ft._modules.get('avgpool')  # Use the model object to select the desired layer
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluation mode

    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    batch_size = 8
    idx = 0
    topk = 5

    while idx < len(images):
        img_tensor = []
        print(idx)
        for img_file in images[idx:idx+batch_size]:
            pil_img = Image.fromarray(np.load(img_file))
            t_img = Variable(normalize(to_tensor(scaler(pil_img))))
            img_tensor.append(t_img)

        img_tensor = torch.stack(img_tensor)
        my_embedding = torch.zeros([img_tensor.shape[0], 2048, 1, 1])
        img_tensor = img_tensor.to(device)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = layer.register_forward_hook(copy_data)
        outputs = model_ft(img_tensor)
        #_, preds = torch.max(outputs, 1)
        _, preds = torch.sort(outputs, 1, descending=True)

        h.remove()
        my_embedding = torch.reshape(my_embedding, [img_tensor.shape[0], 2048, ]).numpy()
        for i, img_file in enumerate(images[idx:idx+batch_size]):
            img_name = img_file.rstrip('.npy').split('/')[-1]
            img_feature_dict[fn2ent[img_name]] = my_embedding[i]
            pred_label[img_name] = preds[i].cpu().numpy().tolist()[:topk]
        idx += 8
    print(len(set(pred_label).intersection(set(img2label))))

    # calculate accuracy
    cnt1, cnt2 = 0, 0
    for img, gt_label in img2label.items():
        if img in pred_label:
            if gt_label == pred_label[img][0]:
                cnt1 += 1
            if gt_label in pred_label[img]:
                cnt2 += 1

    print("h@1: {}, h@5:{}".format(cnt1/len(img2label), cnt2/len(img2label)))

    kg1_ent2id = get_ent2id(cfg.kg1_ent2id_path)
    kg2_ent2id = get_ent2id(cfg.kg2_ent2id_path)
    ent2id = dict(kg1_ent2id, **kg2_ent2id)
    img_feature_dict = {ent2id[k]: v for k, v in img_feature_dict.items()}

    if save_feat:
        with open(save_feat_path, 'wb') as f:
            pickle.dump(img_feature_dict, f)

    if save_pred:
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_label, f)


if __name__ == "__main__":
    split = 'fr_en'  # 'fr_en' or 'ja_en' or 'zh_en'
    cfg = Config(split)
    extract_feature_resnet152(cfg, save_feat=True, save_pred=True)