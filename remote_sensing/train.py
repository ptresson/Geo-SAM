# general
import os
import re
import glob
import sys
import abc
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
from rtree.index import Index, Property
from typing import Any, Dict, List
from collections import OrderedDict
import warnings
import json

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import torch.utils.tensorboard as tensorboard

# torchgeo
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd
import rasterio

# data
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# custom modules
from samplers import RoiGeoSampler
from utils import intersects_with_img, get_intersected_bboxes, prepare_shapefile_dataset
from utils import export_on_map

warnings.filterwarnings("ignore")


def train_one_epoch(
        train_dataloader, 
        test_dataloader, 
        model, 
        loss_f, 
        batch_size,
        gdf,
        target_variable,
        learning_rate = 10**-5,
        verbose=False,
        ):

    optimizer = Adam(model.parameters(), lr=learning_rate)

    list_y_pred = []
    list_y = []
    train_loss = 0
    test_loss = 0
    model.cuda()
    model.train()

    for batch in train_dataloader:
            imgs_tensor = batch['image']
            labels = list(gdf.loc[gdf['bboxes'].isin(batch['bbox'])][target_variable])
            list_y += labels
            labels = torch.tensor(labels)
            # labels = torch.sub(labels,1) # loss function needs index
            labels = labels.cuda().long()

            output = model(imgs_tensor.cuda())
            list_y_pred += list(output.argmax(1).detach().cpu().numpy())
            """
            if len(labels)!=output.shape[0]:
                output = correct_duplicate_roi(batch['bbox'], output)
            """
            
            loss = loss_f(output, labels)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    list_y_pred_val = []
    list_y_val = []

    model.eval()

    for batch in test_dataloader:
        imgs_tensor = batch['image']
        labels = list(gdf.loc[gdf['bboxes'].isin(batch['bbox'])][target_variable])
        with torch.no_grad():
            preds = model(imgs_tensor.cuda())
            """
            if len(labels)!=preds.shape[0]:
                preds = correct_duplicate_roi(batch['bbox'], preds)
            """
            list_y_pred_val += list(preds.argmax(1).detach().cpu().numpy())
            list_y_val += labels
            loss = loss_f(preds, torch.tensor(labels).cuda().type(torch.long))
            test_loss += loss.item()


    report = classification_report(list_y_val, list_y_pred_val, labels=np.unique(list_y_val), output_dict=True)
    matrix = confusion_matrix(list_y_val, list_y_pred_val, labels=np.unique(list_y_val))

    if verbose :
        print(classification_report(list_y_val, list_y_pred_val, labels=np.unique(list_y_val)))
        print(confusion_matrix(list_y_val, list_y_pred_val, labels=np.unique(list_y_val)))
        print(f"Avg loss Train: {train_loss:>8f} \t Avg loss Validation: {test_loss:>8f}")

    train_loss /= len(list_y)
    test_loss /= len(list_y_val)


    return model, train_loss, test_loss, report, matrix

def train_kfolds(
        model, 
        dataset, 
        gdf,
        target_variable = 'C_id',
        nepochs=100,
        batch_size=10,
        K=5,
        shuffle=True,
        run_dir='out/runs/',
        ):


    loss_f = nn.CrossEntropyLoss()

    X = np.array([i for i in range(len(gdf))])
    kf = KFold(n_splits=K, shuffle=shuffle)

    fold_best_train_losses = np.zeros(K)
    fold_best_val_losses = np.zeros(K)
    gdf['fold'] = 0

    for fold, (train_index, test_index) in enumerate(kf.split(X)):

        print(f"Fold {fold}: \n")

        writer = tensorboard.SummaryWriter(f'{run_dir}{fold}')

        gdf.loc[test_index,'fold'] = fold # know which point belongs to which fold if needed

        rois_train = gdf.iloc[train_index]['bboxes']
        rois_test = gdf.iloc[test_index]['bboxes']

        sampler_train = RoiGeoSampler(dataset, size = 224, rois = rois_train)
        sampler_test = RoiGeoSampler(dataset, size = 224, rois = rois_test)

        train_dataloader = DataLoader(
                dataset, 
                sampler=sampler_train, 
                collate_fn=stack_samples, 
                shuffle=False, 
                batch_size = batch_size
                )

        test_dataloader = DataLoader(
                dataset, 
                sampler=sampler_test, 
                collate_fn=stack_samples, 
                shuffle=False, 
                batch_size = batch_size
                )

        training_losses = []
        val_losses = []
        best_loss = 10000 # high value to start with
        best_model = model

        for epoch in range(nepochs):
            print(f"{epoch}/{nepochs}", end="\r")
            model, train_loss, test_loss, report, matrix= train_one_epoch(
                                                    train_dataloader, 
                                                    test_dataloader, 
                                                    model, 
                                                    loss_f, 
                                                    gdf = gdf,
                                                    target_variable = target_variable,
                                                    batch_size=batch_size,
                                                    )

            # update best values if test_loss is better
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model
                f1 = report['macro avg']['f1-score']
                with open(f'{run_dir}/{fold}/report.json', "w") as outfile:
                    json.dump(report,outfile)
                writer.add_scalar('loss/best', best_loss, epoch)
                writer.add_scalar('f1/best', f1, epoch)
                torch.save(model.state_dict(),f"{run_dir}/{fold}/best.pth")

            training_losses.append(train_loss)
            val_losses.append(test_loss)

    export_on_map(
            list(gdf['fold']),
            list(gdf['bboxes']),
            gdf.crs,
            out_path=f"{run_dir}sampling.shp"
            )


if __name__ == "__main__":


    img_dir = "/home/hadrien/Traitement_images/images/base_img"
    filename_glob = "1959.tif"    #"output_*_v2*"
    shp_path = '../../rois/ROI_1959.shp'

    backbone_path = '/home/hadrien/Traitement_images/models/DINOv2/outputs/torchgeo_v2/eval/training_87499/teacher_checkpoint.pth'

    patch_h = 16
    patch_w = 16
    in_chans = 1
    pos_embed_size = 257
    num_classes = 6
    freeze_backbone = True

    batch_size = 10

    pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    model = timm.create_model('vit_large_patch16_224', num_classes=num_classes)
    model.patch_embed.proj = torch.nn.Conv2d(1, 1024,
                kernel_size=(14, 14), stride=(14, 14))


    model.pos_embed = nn.Parameter(torch.tensor(pretrained.pos_embed[:, :pos_embed_size]))

    checkpoint = torch.load(backbone_path)
    d = checkpoint['teacher']
    d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
    model.load_state_dict(d2, strict=False)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name[:4]=="head"
            
    class CustomDataset(RasterDataset):
        filename_glob = filename_glob
        is_image = True

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            T.Resize((patch_h * 14, patch_w * 14)), 
            data_keys=["image"])

    dataset = CustomDataset(img_dir, transforms=transform)
    bb = dataset.index.bounds

    loss_f = nn.CrossEntropyLoss()

    gdf = prepare_shapefile_dataset(
            shp_path, 
            img_dir, 
            filename_glob, 
            dataset,
            # sort_geographicaly=True,
            )
    # C_id numbered 1,2,...,6 whereas we need index at 0,1,..,5
    gdf.C_id = gdf.C_id - 1

    train_kfolds(
            model, 
            dataset, 
            gdf, 
            nepochs = 5,
            batch_size = 42,
            )

