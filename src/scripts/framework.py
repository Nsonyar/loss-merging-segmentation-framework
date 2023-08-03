import os
import PIL
import cv2
import sys
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from datetime import date
from datetime import datetime
from typing import Tuple, List

from loss_dice import DiceLoss
from loss_focal import FocalLossM
from loss_tversky import TverskyLoss
from loss_boundary import BoundaryLoss
from loss_hausdorff import HausdorffLoss

from models.enet.model import *
from models.unet.model import UNet
from argparse import ArgumentParser
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import Dice
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassJaccardIndex

torch.set_float32_matmul_precision('high')


def log_torch_tensor(tensor, func_name, var_name):
    print(f'Func:{func_name}, Variable:{var_name}.shape:', tensor.shape)
    print(
        f'Func:{func_name}, Variable:torch.unique({var_name},return_counts=True):',
        torch.unique(
            tensor,
            return_counts=True))


def log_numpy_array(array, func_name, var_name):
    print(f'Func:{func_name}, Variable:{var_name}.shape:', array.shape)
    print(
        f'Func:{func_name}, Variable:np.unique({var_name},return_counts=True):',
        np.unique(
            array,
            return_counts=True))


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class DATASET(Dataset):
    def __init__(
        self,
        data_path,
        split_ratio,
        split,
        img_size=(256, 256),
        transform=None,
        selection_percentage=1,
        dataset='Medaka',
        num_classes=2
    ):
        self.selection_percentage = selection_percentage
        self.dataset = dataset
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.transform = transform
        self.num_classes = num_classes
        self.img_size = img_size

        if self.split == 'train' or self.split == 'valid':
            self.img_path = os.path.join(
                self.data_path, self.dataset, 'training/labels')
            self.mask_path = os.path.join(
                self.data_path, self.dataset, 'training/masks')
        if self.split == 'test':
            self.img_path = os.path.join(
                self.data_path, self.dataset, 'testing/labels')
            self.mask_path = os.path.join(
                self.data_path, self.dataset, 'testing/masks')

        self.img_list, self.mask_list = self.get_filenames_combined(
            self.img_path, self.mask_path, self.selection_percentage)

        if self.split == 'train' or self.split == 'valid':
            # Split between train and valid set
            random_inst = random.Random(12345)
            n_items = len(self.img_list)
            idxs = random_inst.sample(
                range(n_items), n_items // self.split_ratio)
            if self.split == 'train':
                idxs = [idx for idx in range(n_items) if idx not in idxs]
            self.img_list = [self.img_list[i] for i in idxs]
            self.mask_list = [self.mask_list[i] for i in idxs]

        self.valid_labels = self.check_for_heterogeneity()

        self.class_map_train = dict(
            zip(self.valid_labels, range(len(self.valid_labels))))

        self.class_map_test = dict(
            zip(self.valid_labels, range(len(self.valid_labels))))

    """
    This function checks for heterogeneity in the labels of the masks passed to the object. It raises an exception if labels with different classes are detected or if the number of unique values in the mask is not equal to the number of classes passed to the object.
    Returns:
        tuple: A tuple of unique values and their counts in the mask.
    """

    def check_for_heterogeneity(self):
        for i in range(len(self.mask_list)):
            if(i == 0):
                img = cv2.imread(self.mask_list[i], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(
                    img, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
                unique_first, counts_first = np.unique(img, return_counts=True)
            img = cv2.imread(self.mask_list[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(
                img, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
            unique, counts = np.unique(img, return_counts=True)
            if(np.array_equal(unique, unique_first) == False):
                raise Exception(
                    'Labels with different classes or absent classes detected. Check mask: ' +
                    self.mask_list[i])
            if(len(unique) != self.num_classes):
                raise Exception(
                    'More unique values than number of classes detected. Check following mask: ' +
                    self.mask_list[i])
        return tuple(np.asarray((unique, counts)).T[:, 0])

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        print('Get Item')
        print(idx)
        print(self.img_list[idx])
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))

        mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        mask = self.encode_segmap(mask)

        assert(mask.shape == (self.img_size[0], self.img_size[1]))

        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, self.img_size[0], self.img_size[1]))

        return img, mask

    def encode_segmap(self, mask):
        '''
        Function encodes mask given with classes set in __init__ with int values [0,1,2,3] where 0 stands for the background and the rest as encoding for each class
        '''
        if self.split == 'train' or self.split == 'valid':
            for j in self.valid_labels:
                mask[mask == j] = self.class_map_train[j]
        if self.split == 'test':
            for i in self.valid_labels:
                mask[mask == i] = self.class_map_test[i]
        return mask

    def get_filenames_combined(self,
                               image_path: str,
                               mask_path: str,
                               selection_percentage: float) -> Tuple[List[str],
                                                                     List[str]]:
        """
        This function takes image and mask file path, a percentage as an input, reads all the files in both directories, checks if all the files have the same extension, select a random fraction of images and masks,combines the paths of the images and masks and returns them in a tuple.
        :param image_path: A string representing the path of the directory containing the images.
        :param mask_path: A string representing the path of the directory containing the masks.
        :param selection_percentage: A float value between 0 and 1 representing the percentage of images to be selected
        :return: A tuple containing lists of image and mask file paths
        """
        image_list = list()
        image_list, image_extension = self.get_path_list(image_path)
        mask_list, mask_extension = self.get_path_list(mask_path)
        random.seed(1234567)
        num_elements = len(image_list)
        num_to_select = int(num_elements * selection_percentage)
        image_list_fraction = random.sample(image_list, num_to_select)
        image_path_list = []
        mask_path_list = []
        for img in image_list_fraction:
            image_path_list.append(os.path.join(image_path, img))
            mask_path_list.append(os.path.join(mask_path, img).replace(
                image_extension, mask_extension))
        return image_path_list, mask_path_list

    def get_path_list(self, image_path: str) -> Tuple[List[str], str]:
        """
        This function takes a file path as an input, reads all the files in the directory, checks if all the files have the same extension and returns the list of file names and the extension in a tuple.
        :param image_path: A string representing the path of the directory containing the images.
        :return: A tuple containing a list of file names and the extension of the files.
        :raises Exception: Raises an exception if files with different extensions are found.
        """
        image_list = os.listdir(image_path)
        for i in range(len(image_list)):
            if(i == 0):
                extension = image_list[0][-4:]
            if(image_list[i][-4:] != extension):
                raise Exception("Invalid extension found in file: " +
                                image_list[i] + "Program aborted")
        return image_list, extension


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.data_path = hparams.data_path
        self.batch_size = hparams.batch_size
        self.split_ratio = hparams.split_ratio
        self.dataset = hparams.dataset
        self.num_classes = hparams.num_classes
        self.img_size = hparams.img_size
        self.transform = {'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])}
        self.selection_percentage = hparams.selection_percentage

    def setup(self, stage=None):
        self.trainset = DATASET(
            self.data_path,
            self.split_ratio,
            split='train',
            transform=self.transform['train'],
            selection_percentage=self.selection_percentage,
            dataset=self.dataset,
            num_classes=self.num_classes,
            img_size=self.img_size)
        self.validset = DATASET(
            self.data_path,
            self.split_ratio,
            split='valid',
            transform=self.transform['valid'],
            selection_percentage=self.selection_percentage,
            dataset=self.dataset,
            num_classes=self.num_classes,
            img_size=self.img_size)
        self.testset = DATASET(
            self.data_path,
            self.split_ratio,
            split='test',
            transform=self.transform['test'],
            selection_percentage=self.selection_percentage,
            dataset=self.dataset,
            num_classes=self.num_classes,
            img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16)

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16)

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=1,
            shuffle=False,
            num_workers=16)

    def predict_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=1,
            shuffle=False,
            num_workers=16)


''' Function prints n test predictions on wandb. Note that this function will always use the first n samples from the test set.
Args:
    prediction: list of predictions. Number is equal to samples in testset. Each sample in the list is tuple of y_hat ([N,C,H,W]), y ([N,H,W]) and x ([N,3,H,W])
    num_tests:  Numer of tests
    dataset:    name of current dataset
'''


def print_n_tests(predictions, num_tests, dataset):
    for i in range(num_tests):
        y_hat, y, x = predictions[i]
        # --------------------------------------------------------------------------------
        # Prepare image for export and get prediction
        # --------------------------------------------------------------------------------
        x = x.float().cuda()
        x = torch.squeeze(x, 0)
        unorm = UnNormalize(mean=[0.35675976, 0.37380189, 0.3764753], std=[
            0.32064945, 0.32098866, 0.32325324])
        unorm(x)

        # --------------------------------------------------------------------------------
        # Prepare prediction
        # --------------------------------------------------------------------------------
        y_hat = y_hat.cpu().detach().numpy()
        y_hat = np.argmax(y_hat[0], axis=0)  # (H, W)

        # --------------------------------------------------------------------------------
        # Prepare ground truth
        # --------------------------------------------------------------------------------
        # ([1,H,W]) -> torch.Size([H, W])
        y = torch.squeeze(y, 0)
        # Wandb image logging
        if(dataset == 'Medaka'):
            class_labels = {
                0: "Background",
                1: "Bulbus",
                2: "Atrium",
                3: "Ventricle"
            }
        elif(dataset == 'Melanoma'):
            class_labels = {
                0: "Background",
                1: "Lesion"
            }
        elif(dataset == 'IDRID'):
            class_labels = {
                0: "Background",
                1: "HardExudates",
                2: "Microaneurysms",
                3: "OpticDisc",
                4: "Haemorrhages"
            }
        mask_img = wandb.Image(x, masks={
            "predictions": {
                "mask_data": y_hat,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": y.cpu().detach().numpy(),
                "class_labels": class_labels
            }
        })
        wandb.log({"mask_viewing": mask_img})


class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(
            self,
            num_layers=5,
            learning_rate=0.009,
            features_start=64,
            bilinear=False,
            model_type='UNET',
            loss='Dice',
            auto_learning_rate='False',
            num_classes=4,
            strategy='normal',
            focal_alpha=None,
            focal_gamma=2,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            epochs=1,
            tbm=False,
            gpu=0,
            dataset='Medaka',
            pbm_alpha=2,
            pbm_I=1,
            tbm_alpha=2,
            tbm_beta=2,
            tbm_gamma=2,
            tbm_I_1=1,
            tbm_I_2=1,
            tbm_I_3=1,
            ws_1=1,
            ws_2=1,
            ws_3=0.1,
            nws_1=1,
            nws_2=1,
            nws_3=0.1,
            avg='macro',
            hd_alpha=2.0):
        super().__init__()

        # makes that we can access hyperparamters with self.haparams
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Instantiate loss classes
        self.dice = DiceLoss(do_bg=True)
        self.boundary = BoundaryLoss(do_bg=False, gpu=self.hparams.gpu)
        self.tversky = TverskyLoss(
            do_bg=True, alpha=self.hparams.tversky_alpha, beta=self.hparams.tversky_beta)
        self.hausdorff2 = HausdorffLoss(
            do_bg=False, gpu=self.hparams.gpu, hd_alpha=self.hparams.hd_alpha)
        self.focal = FocalLossM(
            do_bg=True, gamma=self.hparams.focal_gamma, alpha=self.hparams.focal_alpha)

        # Instantiate Softmax normalizer
        self.softmax = nn.Softmax(dim=1)

        # Instantiate Model types
        if(self.hparams.model_type == 'UNET'):
            self.net = UNet(num_classes=self.hparams.num_classes,
                            num_layers=self.hparams.num_layers,
                            features_start=self.hparams.features_start,
                            bilinear=self.hparams.bilinear)
        elif(self.hparams.model_type == "ENET"):
            self.net = ENet(num_classes=self.hparams.num_classes)
        else:
            print('Unsupported model set. Please manually check !')

        self.dice_mc = Dice(
            average=self.hparams.avg, num_classes=self.hparams.num_classes)
        self.accuracy_mc = MulticlassAccuracy(
            average=self.hparams.avg, num_classes=self.hparams.num_classes)
        self.jaccIdx_mc = MulticlassJaccardIndex(
            average=self.hparams.avg, num_classes=self.hparams.num_classes)
        self.jaccIdx_mc_ind = MulticlassJaccardIndex(
            average='none', num_classes=self.hparams.num_classes)
        self.precision_mc = MulticlassPrecision(
            average=self.hparams.avg, num_classes=self.hparams.num_classes)
        self.recall_mc = MulticlassRecall(
            average=self.hparams.avg, num_classes=self.hparams.num_classes)

        # Instantiate gradual annealing variables
        self.gradual_weights = torch.tensor([1, 1, 1], device=self.device)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self(x)
        y_one_hot_mask = self.one_hot_encoder(
            y, y_hat, self.hparams.num_classes)
        loss = self.calculate_loss(y_hat, y_one_hot_mask)
        self.print_loss_and_metrics(loss, y_hat, y_one_hot_mask, 'train')
        if(type(loss) is tuple):
            return loss[0]
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self(x)
        y_one_hot_mask = self.one_hot_encoder(
            y, y_hat, self.hparams.num_classes)
        loss = self.calculate_loss(y_hat, y_one_hot_mask)
        self.print_loss_and_metrics(loss, y_hat, y_one_hot_mask, 'val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self(x)
        y_one_hot_mask = self.one_hot_encoder(
            y, y_hat, self.hparams.num_classes)
        loss = self.calculate_loss(y_hat, y_one_hot_mask)
        self.print_loss_and_metrics(loss, y_hat, y_one_hot_mask, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x), y, x

    def configure_optimizers(self):
        if(self.hparams.auto_learning_rate):
            lr = self.learning_rate
        else:
            lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    ''' Transform sparse mask into one-hot mask
    Args:
        y:  ([N,H,W]) integer encoded ground truth mask
        x:  ([N,C,H,W]) logit output of NN
        class_num: number of different classes
    Return:
        y_one_hot_mask: ([N,C,H,W]) one-hot encoded ground truth mask
    '''

    def one_hot_encoder(self, y, y_hat, num_classes):
        y_shape = tuple(y.shape)  # (N, H, W, ...)
        new_shape = (y_shape[0], num_classes) + y_shape[1:]  # (N,C,H,W)
        one_hot = torch.zeros(new_shape)  # (N,C,H,W)
        one_hot = one_hot.to(y_hat)
        index_tensor = y.unsqueeze(1)  # (N,1,H,W)
        dim = 1
        src = 1
        y_one_hot_mask = one_hot.scatter_(dim, index_tensor, src)  # (N,C,H,W)
        return y_one_hot_mask

    ''' Function logs loss function and all metrics
    Args:
        loss:   tuple or current loss value. Tuple contains loss, list of combined loss values and list of combined loss names
        x:      predictions
        y:      ground truth
        step:   training step (train,validation,test)
    '''

    def print_loss_and_metrics(self, loss, y_hat, y_one_hot_mask, step):
        y_hat_softmax = self.softmax(y_hat)
        y_argmax = torch.argmax(y_one_hot_mask, dim=1)
        if(type(loss) is tuple):
            self.log(step + '_loss', loss[0], prog_bar=True)
            for i in range(len(loss[1])):
                if(loss[2][i] == 'CE' or loss[2][i] == 'Focal'):
                    self.log(step + '_loss_db_'+loss[2][i], loss[1][i])
                elif(loss[2][i] == 'Dice' or loss[2][i] == 'Tversky'):
                    self.log(step + '_loss_rb_'+loss[2][i], loss[1][i])
                elif(loss[2][i] == 'Boundary' or loss[2][i] == 'HD'):
                    self.log(step + '_loss_bb_'+loss[2][i], loss[1][i])
                else:
                    raise Exception('Wrong loss name provided')
        else:
            self.log(step + '_loss', loss)
        self.log(step + '_dice', self.dice_mc(y_hat_softmax, y_argmax))
        self.log(step + '_accuracy', self.accuracy_mc(y_hat_softmax, y_argmax))
        self.log(step + '_jaccIdx', self.jaccIdx_mc(y_hat_softmax, y_argmax))
        self.log(
            step + '_precision',
            self.precision_mc(
                y_hat_softmax,
                y_argmax))
        self.log(step + '_recall_mc', self.recall_mc(y_hat_softmax, y_argmax))

        jaccIdx_mc_ind = self.jaccIdx_mc_ind(y_hat_softmax, y_argmax)

        if(self.hparams.dataset == 'IDRID'):
            classes = ['Background', 'HardExudates',
                       'Microaneurysms', 'OpticDisc', 'Haemorrhages']
        elif(self.hparams.dataset == 'Melanoma'):
            classes = ['Background', 'Lesion']
        elif(self.hparams.dataset == 'Medaka'):
            classes = ['Background', 'Bulbus', 'Atrium', 'Ventricle']
        else:
            raise Exception(
                'Wrong dataset setup in parameters. Manually check !')

        if(len(jaccIdx_mc_ind) == len(classes)):
            for i in range(len(classes)):
                self.log(step + '_jaccIdx_' +
                         classes[i], jaccIdx_mc_ind[i].item())
        else:
            raise Exception(
                'Length of classes and predictions divergent. Manually check !')

    def calculate_loss(self, y_hat, y_one_hot_mask):
        y_hat_softmax = self.softmax(y_hat)
        if('_' in self.hparams.loss):
            loss_names_list = self.hparams.loss.split('_')
            loss_list = []
            for i in range(len(loss_names_list)):
                if(loss_names_list[i] == 'CE'):
                    loss_list.append(F.cross_entropy(y_hat, y_one_hot_mask))
                elif(loss_names_list[i] == 'Focal'):
                    loss_list.append(self.focal(y_hat, y_one_hot_mask))
                elif(loss_names_list[i] == 'Tversky'):
                    loss_list.append(self.tversky(
                        y_hat_softmax, y_one_hot_mask))
                elif(loss_names_list[i] == 'Dice'):
                    loss_list.append(self.dice(y_hat_softmax, y_one_hot_mask))
                elif(loss_names_list[i] == 'HD'):
                    loss_list.append(self.hausdorff2(
                        y_hat_softmax, y_one_hot_mask))
                elif(loss_names_list[i] == 'Boundary'):
                    loss_list.append(self.boundary(
                        y_hat_softmax, y_one_hot_mask))
                else:
                    print('Wrong loss provided. Please manually check !')

            return self.strategy(loss_list, loss_names_list), loss_list, loss_names_list
        else:
            if(self.hparams.loss == 'CE'):
                loss = F.cross_entropy(y_hat, y_one_hot_mask)
            elif(self.hparams.loss == 'Focal'):
                loss = self.focal(y_hat, y_one_hot_mask)
            elif(self.hparams.loss == 'Tversky'):
                loss = self.tversky(y_hat_softmax, y_one_hot_mask)
            elif(self.hparams.loss == 'Dice'):
                loss = self.dice(y_hat_softmax, y_one_hot_mask)
            elif(self.hparams.loss == 'HD'):
                loss = self.hausdorff2(y_hat_softmax, y_one_hot_mask)
            elif(self.hparams.loss == 'Boundary'):
                loss = self.boundary(y_hat_softmax, y_one_hot_mask)
            else:
                print('Wrong loss provided. Please manually check !')
            return loss

    def strategy(self, loss_list, loss_names_list):
        if(self.hparams.strategy == 'max_strategy'):
            return self.max_strategy(loss_list)
        elif(self.hparams.strategy == 'min_strategy'):
            return self.min_strategy(loss_list)
        elif(self.hparams.strategy == 'arithmetic'):
            return self.arithmetic(loss_list)
        elif(self.hparams.strategy == 'harmonic'):
            return self.harmonic(loss_list)
        elif(self.hparams.strategy == 'weighted_sum'):
            return self.weighted_sum(loss_list, loss_names_list)
        elif(self.hparams.strategy == 'normalized_weighted_sum'):
            return self.normalized_weighted_sum(loss_list, loss_names_list)
        elif(self.hparams.strategy == 'performance_based_merging'):
            return self.performance_based_merging(loss_list)
        else:
            raise Exception(
                'Wrong strategy provided. Select correct description. Script terminated !')

    def max_strategy(self, losses):
        """
        Finds the maximum value in the input list of losses.

        Args:
            losses (list): A list of losses.

        Returns:
            float: The maximum value in the list of losses.
        """
        weighted_loss_list = [loss * gradual_weight for loss,
                              gradual_weight in zip(losses, self.gradual_weights)]
        return max(weighted_loss_list)

    def min_strategy(self, losses):
        """
        Finds the minimum value in the input list of losses.

        Args:
            losses (list): A list of losses.

        Returns:
            float: The minimum value in the list of losses.
        """
        weighted_loss_list = [loss * gradual_weight for loss,
                              gradual_weight in zip(losses, self.gradual_weights)]
        return min(weighted_loss_list)

    def arithmetic(self, losses):
        """
        Calculates the mean (average) of the input list of losses.

        Args:
            losses (list): A list of losses.

        Returns:
            float: The mean of the input list of losses.
        """
        weighted_loss_list = [loss * gradual_weight for loss,
                              gradual_weight in zip(losses, self.gradual_weights)]
        return sum(weighted_loss_list) / len(losses)

    def harmonic(self, losses):
        """
        Calculates the harmonic mean of the input list of losses.

        Args:
            losses (list): A list of losses.

        Returns:
            float: The harmonic mean of the input list of losses.
        """
        # Invert the losses and calculate their mean
        inverted_losses = [1 / loss for loss in losses]
        mean_of_inverted_losses = self.arithmetic(inverted_losses)

        # Invert the mean of the inverted losses to get the harmonic mean
        return 1 / mean_of_inverted_losses

    def weighted_sum(self, losses, loss_names_list):
        """
        Calculates a weighted sum of the input losses.

        Args:
            losses (list): A list of losses.
            a (float): The weight to apply to the first loss. Defaults to 1.
            b (float): The weight to apply to the second loss. Defaults to 1.
            c (float): The weight to apply to the third loss (if present). Defaults to 0.1.

        Returns:
            float: The weighted sum of the input losses.
        """
        number_losses = len(losses)
        if number_losses == 2:
            # Here we make sure to weight boundary based losses with 0.1
            if(loss_names_list[0] == 'HD' or loss_names_list[0] == 'Boundary'):
                a = 0.1
                b = 1
            elif(loss_names_list[1] == 'HD' or loss_names_list[1] == 'Boundary'):
                a = 1
                b = 0.1
            else:
                a = 1
                b = 1
            weights = [a, b]
        elif number_losses == 3:
            weights = [self.hparams.ws_1, self.hparams.ws_2, self.hparams.ws_3]
        else:
            raise ValueError(
                "Expected 2 or 3 losses, got {}".format(number_losses))

        # Calculate annealed loss
        weighted_loss = sum(weight * loss * gradual_weight for weight, loss,
                            gradual_weight in zip(weights, losses, self.gradual_weights))

        return weighted_loss

    def normalized_weighted_sum(self, losses, loss_names_list):
        """
        Calculates a normalized weighted sum of the input losses.

        Args:
            losses (list): A list of losses.
            a (float): The weight to apply to the first loss. Defaults to 1.
            b (float): The weight to apply to the second loss. Defaults to 1.
            c (float): The weight to apply to the third loss (if present). Defaults to 0.1.

        Returns:
            float: The normalized weighted sum of the input losses.
        """
        number_losses = len(losses)
        if number_losses == 2:
            # Here we make sure to weight boundary based losses with 0.1
            if(loss_names_list[0] == 'HD' or loss_names_list[0] == 'Boundary'):
                a = 0.1
                b = 1
            elif(loss_names_list[1] == 'HD' or loss_names_list[1] == 'Boundary'):
                a = 1
                b = 0.1
            else:
                a = 1
                b = 1
            weights = [a, b]
        elif number_losses == 3:
            weights = [self.hparams.nws_1,
                       self.hparams.nws_2, self.hparams.nws_3]
        else:
            raise ValueError(
                "Expected 2 or 3 losses, got {}".format(number_losses))

        # Calculate the normalized weights
        normalized_weights = [w / sum(weights) for w in weights]

        weighted_loss = sum(weight * loss * gradual_weight for weight, loss,
                            gradual_weight in zip(normalized_weights, losses, self.gradual_weights))

        return weighted_loss

    def performance_based_merging(self, losses):
        alpha = self.hparams.pbm_alpha
        I = self.hparams.pbm_I
        norm = [loss / sum(losses) for loss in losses]
        weights = [(1-I)*n ** alpha + I * (1-n)**alpha for n in norm]
        # Calculate annealed loss
        annealed_loss = sum(weight * loss * gradual_weight for weight, loss,
                            gradual_weight in zip(weights, losses, self.gradual_weights))
        # print(annealed_loss)
        return annealed_loss

    def on_validation_epoch_end(self):
        alpha = self.hparams.tbm_alpha
        beta = self.hparams.tbm_beta
        gamma = self.hparams.tbm_gamma
        tbm_I_1 = self.hparams.tbm_I_1
        tbm_I_2 = self.hparams.tbm_I_2
        tbm_I_3 = self.hparams.tbm_I_3

        # if not (alpha >= 1 and beta >= 1 and gamma >= 1):
        #     raise ValueError(
        #         "alpha, beta, and gamma must be equal to or greater than 1")

        # if not (tbm_I_1 == 0 or tbm_I_1 == 1) or not (tbm_I_2 == 0 or tbm_I_2 == 1) or not (tbm_I_3 == 0 or tbm_I_3 == 1):
        #     raise ValueError(
        #         "tbm_I_1, tbm_I_2, and tbm_I_3 must be either 0 or 1")

        if not self.hparams.tbm:
            # Update weights based on the current epoch number and whether double or triple combination used
            if(len(self.hparams.loss.split('_')) == 3):
                self.gradual_weights = torch.tensor([1, 1, 1])
            elif(len(self.hparams.loss.split('_')) == 2):
                self.gradual_weights = torch.tensor([1, 1])
            else:
                self.gradual_weights = torch.tensor([1])
        else:
            # Update weights based on the current epoch number
            current_epoch = self.current_epoch
            epoch_ratio = current_epoch / self.hparams.epochs
            if(len(self.hparams.loss.split('_')) == 3):
                self.gradual_weights = torch.tensor([(1 - tbm_I_1) * epoch_ratio ** alpha + tbm_I_1 * (1 - epoch_ratio)**alpha, (1 - tbm_I_2) *
                                                    epoch_ratio ** beta + tbm_I_2 * (1 - epoch_ratio)**beta, (1 - tbm_I_3) * epoch_ratio ** gamma + tbm_I_3 * (1 - epoch_ratio)**gamma])
            elif(len(self.hparams.loss.split('_')) == 2):
                self.gradual_weights = torch.tensor([(1 - tbm_I_1) * epoch_ratio ** alpha + tbm_I_1 * (
                    1 - epoch_ratio)**alpha, (1 - tbm_I_2) * epoch_ratio ** beta + tbm_I_2 * (1 - epoch_ratio)**beta])
            else:
                self.gradual_weights = torch.tensor(
                    [(1 - tbm_I_1) * epoch_ratio ** alpha + tbm_I_1 * (1 - epoch_ratio)**alpha])


class LossMerging():
    def __init__(self) -> None:
        # Loss functions and types
        self.loss_dict = {
            1: {'type': 'db', 'name': 'CE'},
            2: {'type': 'db', 'name': 'Focal'},
            3: {'type': 'rb', 'name': 'Tversky'},
            4: {'type': 'rb', 'name': 'Dice'},
            5: {'type': 'bb', 'name': 'HD'},
            6: {'type': 'bb', 'name': 'Boundary'},
        }
        # Loss Combinations
        self.loss_combinations = self.get_loss_combinations(
            self.loss_dict, triple=True, double=True, baseline=False)
        print(self.loss_combinations)
        #Merge Strategies
        self.merge_strategies=['performance_based_merging']
        #Configuration
        self.sweep_configuration = {
            'method': 'random',
            'name': 'top',
            'metric': {'goal': 'minimize', 'name': 'val_loss'},
            'parameters':
            {   
                'strategy': {'values': self.merge_strategies},
                'selection_percentage':{'values':[1.0,0.64,0.32]},
                'loss': {'values': self.loss_combinations},
            }
        }

    ''' Function combines all losses from each loss type
    Args:
        loss_dict:          dictionary with names of loss functions sorted by type
        include_standard:   flat whether to include standard individual functions
    '''

    def get_loss_combinations(self, loss_dict, triple=False, double=False, baseline=False):
        self.check_number_of_loss_per_type(loss_dict)
        db = []
        rb = []
        bb = []
        final_list = []
        ''' 
        ['CE', 'Focal']
        ['Tversky', 'Dice']
        ['HD', 'Boundary']
        '''
        for k_1, v_1 in loss_dict.items():
            if(v_1['type'] == 'db'):
                db.append(v_1['name'])
        for k_2, v_2 in loss_dict.items():
            if(v_2['type'] == 'rb'):
                rb.append(v_2['name'])
        for k_3, v_3 in loss_dict.items():
            if(v_3['type'] == 'bb'):
                bb.append(v_3['name'])

        # Append solo combinations
        if(baseline == True):
            result = []
            result_solo = []
            for h in range(len(db)):
                result.append(db[h])
                result.append(rb[h])
                result.append(bb[h])
            final_list.extend(result)

        # Append double combinations
        if(double == True):
            result_double = []
            result = []
            # Append combinations of 2
            for a in range(len(db)):
                for b in range(len(rb)):
                    result.append(db[a])
                    result.append(rb[b])

            for a in range(len(rb)):
                for b in range(len(bb)):
                    result.append(rb[a])
                    result.append(bb[b])

            for a in range(len(bb)):
                for b in range(len(db)):
                    result.append(bb[a])
                    result.append(db[b])

            for i in range(0, len(result), 2):
                result_double.append(result[i] + '_' + result[i + 1])
            final_list.extend(result_double)

        # Append triple combinations
        if(triple == True):
            result_triple = []
            result = []
            for i in range(len(db)):
                for j in range(len(rb)):
                    for k in range(len(bb)):
                        result.append(db[i])
                        result.append(rb[j])
                        result.append(bb[k])

            for i in range(0, len(result), 3):
                result_triple.append(
                    result[i] + '_' + result[i + 1] + '_' + result[i + 2])

            final_list.extend(result_triple)

        if(len(final_list) == 0):
            raise Exception('No loss combination selected. Program aborting !')
        return final_list

    def check_number_of_loss_per_type(self, loss_dict):
        counter_db = 0
        counter_rb = 0
        counter_bb = 0
        for k, v in loss_dict.items():
            if(v['type'] == 'db'):
                counter_db = counter_db + 1
            if(v['type'] == 'rb'):
                counter_rb = counter_rb + 1
            if(v['type'] == 'bb'):
                counter_bb = counter_bb + 1
        if(counter_db == counter_rb == counter_bb):
            print('Equal number of losses available. Script proceeding !')
        else:
            raise Exception(
                'Number of loss functions not equal per type in variable loss_dict')

    def main(self):
        hyperparameter_defaults = dict(
            #Checkpoint and model saving ------------------------------------------------#
            save_top_k=1,
            checkpoint_available=True,
            #Data and evaluation --------------------------------------------------------#
            num_tests=20,
            split_ratio=5,
            avg='macro',
            selection_percentage=1,
            img_size=(512,512),
            data_path='data_semantics',
            dataset='IDRID',
            #Hardware and infrastructure ------------------------------------------------#
            gpu=0,
            #Loss function --------------------------------------------------------------#
            tversky_alpha=0.3,
            tversky_beta=0.7,
            focal_alpha=None,
            focal_gamma=2,
            hd_alpha=2.0,
            loss='Dice',
            #Merge strategy -------------------------------------------------------------#
            ws_1=1.0,
            ws_2=1.0,
            ws_3=0.1,
            nws_1=1,
            nws_2=1,
            nws_3=0.1,
            pbm_alpha=4.4,
            pbm_I=0.76,
            tbm=False,
            tbm_alpha=2,
            tbm_beta=2,
            tbm_gamma=2,
            tbm_I_1=1,
            tbm_I_2=1,
            tbm_I_3=1,
            strategy='performance_based_merging',
            #Model architecture ---------------------------------------------------------#
            num_layers=5,
            features_start=64,
            bilinear=False,
            model_type="UNET",
            #Training configuration -----------------------------------------------------#
            batch_size=1,
            learning_rate=0.009,
            grad_batches=1,
            epochs=100,
            auto_learning_rate=False,
            early_stopping=False,
            #Transfer learning configuration --------------------------------------------#
            transfer=True
        )

        # We can add mode="offline" if we want to run locally
        run = wandb.init(config=hyperparameter_defaults)
        config = wandb.config
        print(run)
        print(run.name)

        # Check if dataset was written correctly and update proper number of
        # classes. These classes need to be given manually and are later verified
        # for validity
        if(config.dataset == 'Medaka'):
            wandb.config.update({'num_classes': 4}, allow_val_change=True)
        elif(config.dataset == 'Melanoma'):
            wandb.config.update({'num_classes': 2}, allow_val_change=True)
        elif(config.dataset == 'IDRID'):
            wandb.config.update({'num_classes': 5}, allow_val_change=True)
        else:
            raise Exception('Provide a valid dataset name. Program aborted')

        print(f'Starting run {run.name} with configurations as {config}')

        #Checkpointing
        if(config.checkpoint_available):
            # Insert checkpoint here from wandb.ai artifacts
            artifact = run.use_artifact(
                '<sweep username>/<sweep name>/<model name>:<model version>', type='model')
            artifact_dir = artifact.download()
        else:
            artifact_dir = None

        # --------------------------------------------------------------------------------
        # 1 LIGHTNING MODEL
        # --------------------------------------------------------------------------------
        if(artifact_dir is None):
            model = SegModel(num_layers=config.num_layers,
                             learning_rate=config.learning_rate,
                             features_start=config.features_start,
                             bilinear=config.bilinear,
                             model_type=config.model_type,
                             loss=config.loss,
                             auto_learning_rate=config.auto_learning_rate,
                             num_classes=config.num_classes,
                             strategy=config.strategy,
                             focal_alpha=config.focal_alpha,
                             focal_gamma=config.focal_gamma,
                             tversky_alpha=config.tversky_alpha,
                             tversky_beta=config.tversky_beta,
                             epochs=config.epochs,
                             tbm=config.tbm,
                             gpu=config.gpu,
                             dataset=config.dataset,
                             pbm_alpha=config.pbm_alpha,
                             pbm_I=config.pbm_I,
                             tbm_alpha=config.tbm_alpha,
                             tbm_beta=config.tbm_beta,
                             tbm_gamma=config.tbm_gamma,
                             tbm_I_1=config.tbm_I_1,
                             tbm_I_2=config.tbm_I_2,
                             tbm_I_3=config.tbm_I_3,
                             ws_1=config.ws_1,
                             ws_2=config.ws_2,
                             ws_3=config.ws_3,
                             nws_1=config.nws_1,
                             nws_2=config.nws_2,
                             nws_3=config.nws_3,
                             avg=config.avg,
                             hd_alpha=config.hd_alpha)

        else:
            model = SegModel.load_from_checkpoint(
                Path(artifact_dir) / "model.ckpt")


        # ----------------------------------------------------------------------------
        # 2 DATA PIPELINES
        # ----------------------------------------------------------------------------
        dataModule = DataModule(config)

        # ----------------------------------------------------------------------------
        # 3 WANDB LOGGER
        # ----------------------------------------------------------------------------
        wandb_logger = WandbLogger(log_model=True)
        wandb_logger.watch(model.net)
        csv_logger = CSVLogger("logs_csv", name=run.name)

        # ----------------------------------------------------------------------------
        # 4 DEFINE CALLBACKS
        # ----------------------------------------------------------------------------
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_weights_only=True)

        callbacks = [model_checkpoint_callback]

        if(config.early_stopping):
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1,
                patience=3,
                verbose=True,
                mode="min"
            )
            callbacks.append(early_stop_callback)
        # ----------------------------------------------------------------------------
        # 5 TRAINER
        # ----------------------------------------------------------------------------
        trainer = pl.Trainer(
            # auto_lr_find=False,
            accelerator='gpu',
            devices=[config.gpu],
            logger=[wandb_logger, csv_logger],
            max_epochs=config.epochs,
            accumulate_grad_batches=config.grad_batches,
            callbacks=callbacks,
        )
        # ----------------------------------------------------------------------------
        # 6 FIND LEARNING RATE
        # ----------------------------------------------------------------------------
        if(config.auto_learning_rate):
            lr_finder = trainer.tuner.lr_find(model, datamodule=dataModule)
            new_lr = lr_finder.suggestion()
            model.learning_rate = new_lr
            print('New learning rate:', model.learning_rate)
            wandb.config.update({'learning_rate': new_lr},
                                allow_val_change=True)
            
        if(config.transfer==False):
            # ----------------------------------------------------------------------------
            # 7 START TRAINING
            # ----------------------------------------------------------------------------
            trainer.fit(model, dataModule)

            # ----------------------------------------------------------------------------
            # 8 START TESTING
            # ----------------------------------------------------------------------------
            if(config.save_top_k == 0):
                trainer.test(model, datamodule=dataModule)
                predictions = trainer.predict(
                    model, datamodule=dataModule)
            else:
                trainer.test(model, datamodule=dataModule, ckpt_path='best')
                predictions = trainer.predict(
                    model, datamodule=dataModule, ckpt_path='best')
            print_n_tests(predictions, config.num_tests, config.dataset)
            wandb.finish()
        else:
            predictions = trainer.predict(model, datamodule=dataModule)
            print_n_tests(predictions, config.num_tests, config.dataset)
            wandb.finish()



if __name__ == '__main__':

    lossMerging = LossMerging()

    sweep = False
    now = datetime.now()

    if(sweep):
        sweep_id = wandb.sweep(sweep=lossMerging.sweep_configuration,
                               project='sweep_' + str(now).replace(':',
                                                                   '-').replace(' ',
                                                                                '_'))
        wandb.agent(sweep_id, function=lossMerging.main)
    else:
        lossMerging.main()
