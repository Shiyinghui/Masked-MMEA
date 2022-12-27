from __future__ import print_function
from __future__ import division
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time

import copy
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from config import Config
from tools import *


class DB15kImageDataset(Dataset):
    def __init__(self, idx_dir: str, img_dir: str, split: str, mode, input_size):
        super(DB15kImageDataset, self).__init__()
        with open(os.path.join(idx_dir, split+f'_{mode}.pkl'), 'rb') as f:
            info = pickle.load(f)
        self.split = split
        self.mode = mode  # item: (ent, type, img_name, label_id)
        self.images = [os.path.join(img_dir, item[2]) for item in info]
        self.labels = [item[3] for item in info]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        print(max(self.labels), min(self.labels), len(set(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.data_transforms[self.mode](Image.fromarray(np.load(self.images[index])))
        label = self.labels[index]
        return img, label


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def train_model(model, dataloaders, criterion, optimizer, device, lang,
                cfg, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if epoch < int(num_epochs/2):
                    continue
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            iter_cnt = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if iter_cnt % 100 == 0:
                    print('{}: epoch {}, iter {}'.format(phase, epoch, iter_cnt))
                iter_cnt += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                #  forward， track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), f'./resnet_{lang}.pkl')
    save_model(model, cfg.model_path)
    return model, val_acc_history


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # """ Resnet18
        # """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        """ Resnet152
                """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        #model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def main(cfg):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    lang = cfg.ds  # 'fr_en' or 'ja_en' or 'zh_en'
    img_idx_dir = cfg.img_idx_dir
    img_data_dir = cfg.img_data_dir

    model_name = "resnet"
    num_classes = cfg.finetune_classes
    batch_size = 32
    num_epochs = 25   # 15
    feature_extract = False  # True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    #print(model_ft)

    print("Initializing Datasets and Dataloaders...")
    train_dataset = DB15kImageDataset(img_idx_dir, img_data_dir, lang, 'train', input_size)
    test_dataset = DB15kImageDataset(img_idx_dir, img_data_dir, lang, 'test', input_size)

    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True,
                          collate_fn=dataset_collate)
    test_dl = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True,
                         collate_fn=dataset_collate)
    dataloaders_dict = {'train': train_dl, 'test': test_dl}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, device, lang, cfg,
                                 num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))


if __name__ == "__main__":
    split = 'fr_en'  # 'fr_en' or 'ja_en' or 'zh_en'
    cfg = Config(split)
    main(cfg)

