# general
from operator import truediv
import os
import re
import glob
import sys
import abc
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
from rtree.index import Index, Property
from typing import Any, Dict, List
import warnings
from collections import OrderedDict
import joblib
from scipy.stats import fit

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
from kornia.enhance.normalize import Normalize
import timm

# torchgeo
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd
import rasterio
from rasterio.transform import Affine


# data
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import KMeans
import hdbscan
warnings.filterwarnings("ignore")

# custom modules
from samplers import RoiGeoSampler
from utils import get_mean_sd_by_band, intersects_with_img, get_intersected_bboxes, plot_pred_obs, prepare_shapefile_dataset, prepare_shapefile_dataset_cont
from utils import vit_first_layer_with_nchan, reconstruct_img, resize_image, crop_duplicate, load_pretrained_vit
from utils import numpy_to_geotif
from utils import export_on_map
from utils import prepare_shapefile_dataset, correct_duplicate_roi, stack_samples, load_pretrained_vit_and_add_custom_head
from train_utils import check_model_hash, do_train_cont, K_fold, plot_loss_by_fold
from models import SimpleFCHead
from datasets import ROIDataset
from custom_datasets import *

def train_supervised(
        img_dir,
        gdf,
        model,
        loss_f,
        filename_glob = "*",
        transforms = None,
        bboxes_col_name = 'bboxes',
        target_variable='C_id',
        do_K_fold = True,
        test_prop = 0.2,
        batch_size = 10, 
        N_epochs = 100,
        lr = 10**-5,
        model_save_path = "out/proj.pkl",
        ):
    
    if model_save_path[-4:] == '.pkl':
        len_fn = len(model_save_path.split('/')[-1])
        out_dir = model_save_path[:-len_fn-1]
    else :
        out_dir = model_save_path

    if do_K_fold :

        K = int(1/test_prop)

        res, losses = K_fold(
                K, 
                gdf, 
                dataset, 
                model, 
                target_variable=target_variable,
                loss_f=loss_f, 
                bboxes_col_name = bboxes_col_name, 
                batch_size=batch_size, 
                N_epochs=N_epochs, 
                lr=lr, 
                shuffle_folds=True, 
                out_dir=out_dir,
                )

        df = pd.DataFrame(res)

        merged_df = gdf.merge(df, on='bboxes')
        del merged_df['bboxes'] ## class is not understood by fiona ?
        merged_df.to_file(os.path.join(out_dir, f'{target_variable}.shp'))

        # Calculate RMSE by fold
        merged_df['SE'] = (merged_df['predicted'] - merged_df['observed']) ** 2
        rmse_by_fold = merged_df.groupby('folds')['SE'].mean() ** 0.5
        print(len(merged_df['predicted']))
        print(merged_df['folds'])

        # Compute mean RMSE and SD of RMSE values
        mean_rmse = rmse_by_fold.mean()
        sd_rmse = rmse_by_fold.std()

        print(f"RMSE by Fold: {rmse_by_fold}")
        print(f"Mean RMSE = {mean_rmse:.2f} ± {sd_rmse:.2f}")


        plot_pred_obs(
            predicted = res['predicted'], 
            observed = res['observed'],
            out_name = os.path.join(out_dir, f'pred_obs_{target_variable}.png'),
            text_note=f"RMSE = {mean_rmse:.2f} ± {sd_rmse:.2f}",
            )

        plot_loss_by_fold(
                losses, 
                out_name = os.path.join(out_dir, f'losses_{target_variable}.png'),
                )

        
    else:
        train_dataset = ROIDataset(root = img_dir,
                                   transforms = transforms,
                                   gdf = gdf,
                                   target_var = target_variable,)  
        
        test_dataset = ROIDataset(root = img_dir,
                                  transforms = transforms,
                                  gdf = gdf,
                                  target_var = target_variable,) 
          
        train_dataset.filename_glob = filename_glob
        test_dataset.filename_glob = filename_glob

        train_indexes, test_indexes = train_test_split(np.array([i for i in range(len(train_dataset.gdf))]), test_size = test_prop)
        train_dataset.gdf = train_dataset.gdf.iloc[train_indexes]
        test_dataset.gdf = test_dataset.gdf.iloc[test_indexes]

        training_losses = []
        val_losses = []
 
        train_dataloader = DataLoader(train_dataset, collate_fn=stack_samples, shuffle=False, batch_size = batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=stack_samples, shuffle=False, batch_size = batch_size)

        for i in range(N_epochs):
            print(f"\nEpoch {i+1}")
            res = do_train_cont(
                    train_dataloader, 
                    test_dataloader, 
                    model=model, 
                    loss_f=loss_f, 
                    learning_rate=lr,
                    target_variable=target_variable,
                    )
            training_losses.append(res['train_loss'])
            val_losses.append(res['valid_loss'])
        final_model = res['model']
        bboxes = res['bboxes']
        og_labels = res["gt"]
        torch.save(final_model, model_save_path)
        plt.plot(np.arange(1, N_epochs+1, 1), np.array(training_losses), label = "Train loss")
        plt.plot(np.arange(1, N_epochs+1, 1), np.array(val_losses), label = "Validation loss")
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'losses.png'))   


if __name__ == "__main__":
    
    img_glob = '*.[Tt][Ii][Ff]*'
    img_dir = '/home/ptresson/maille/ADMaille/'
    rois_path = '/home/ptresson/maille/points/transfo.shp'

    # checkpoint_path = '/home/hadrien/Traitement_images/models/DINOv2/outputs/torchgeo_v2/eval/training_87499/teacher_checkpoint.pth'
    checkpoint_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_w = 16
    patch_h = 16
    out_classes = 1
    maille_path = '/home/ptresson/maille/'
    admaille_path = '/home/ptresson/maille/ADMaille/'
    sar_path = '/home/ptresson/maille/ADMaille/SAR/'


    ######################## get means and sds of all bands ################
    # tif_file = [os.path.join(admaille_path, f) for f in sorted(os.listdir(admaille_path)) if f.endswith('.tif')]
    # spot_file = [os.path.join(maille_path, f) for f in os.listdir(maille_path) if f.endswith('.tif')]
    # sar_file = [os.path.join(sar_path, f) for f in os.listdir(sar_path) if f.endswith('.tif')]
    # files = spot_file + sar_file + tif_file 
    # files = sar_file + tif_file 
    # print(files)

    # MEANS = []
    # SDS = []

    # for file in files :
    #     mean, sd = get_mean_sd_by_band(file)
    #     MEANS.extend(mean)
    #     print(MEANS)
    #     SDS.extend(sd)
    #     print(SDS)

    # print(MEANS)
    # print(SDS)
    # sys.exit(1)
    ############################################################################


    MEANS = [124.19, 87.61, 47.60, 84.60267379987175, 186.40649016850378, -73.23005676269531, 5036.104005604582, 16376.4873046875, 874.5887982478903, -21.899221420288086, -0.04424861477581546, 0.04424861477581546, -16.967143416587476]
    SDS = [74.40, 53.19, 28.28, 51.12276843481377, 129.5628533539407, 2842.29248046875, 1404.8407341782038, 145635.28125, 162.45557874845173, 2844.227783203125, 0.03822571657935556, 0.03822571657935556, 526.4417166697268]
    # MEANS = [124.19, 87.61, 47.60]
    # SDS = [74.40, 53.19, 28.28]

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(mean=MEANS, std=SDS),
            K.RandomVerticalFlip(0.5),
            K.RandomHorizontalFlip(0.5),
            T.Resize((patch_h * 14, patch_w * 14)),
            data_keys=["image"]
            )

    ## combine different sources
    spot = Spot_maille(maille_path)
    sar = SAR_maille(sar_path)

    bi = BI(admaille_path)
    catchment_area = CA(admaille_path)
    dem_zone_reproj = DEM(admaille_path)
    ls_factor = LS(admaille_path)
    ndsi = NDSI(admaille_path)
    ndvi = NDVI(admaille_path)
    slope = Slope(admaille_path)
    TWI = TWI_maille(admaille_path)


    ## spectral
    dataset = spot
    dataset = dataset & sar

    ## ancillary variables
    dataset = dataset & bi
    dataset = dataset & catchment_area
    dataset = dataset & dem_zone_reproj
    dataset = dataset & ls_factor
    dataset = dataset & ndsi
    dataset = dataset & ndvi
    dataset = dataset & slope
    dataset = dataset & TWI

    dataset.transforms = transform

    # model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model = load_pretrained_vit_and_add_custom_head(
            checkpoint_path, 
            in_chans=len(MEANS), 
            out_classes=out_classes, 
            freeze_backbone = True,
            freeze_proj = False,
            )

    model.to(device)

    for target_variable in ["pH_H2O", "pH_KCl", "pot_acid", "Si"]:

        gdf = prepare_shapefile_dataset_cont(
                rois_path, 
                img_dir, 
                img_glob, 
                dataset,
                target_variable=target_variable
                )

        train_supervised(
                dataset, 
                model, 
                gdf, 
                do_K_fold = True, 
                lr=10**-3,
                test_prop = 0.2, 
                batch_size=10,
                N_epochs=50, 
                loss_f=nn.MSELoss(),
                target_variable=target_variable,
                model_save_path = '../logs/supervised/'
                )
