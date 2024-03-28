import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import KFold
from torch.optim import Adam, optimizer
from utils import DynamicFCHead, EarlyStopper, FCHead, ThreeFCHead, load_pretrained_vit_and_add_custom_head, prepare_shapefile_dataset_cont, remove_black_tiles, correct_duplicate_roi, stack_samples
from samplers import RoiGeoSampler
from copy import deepcopy
import pandas as pd
from utils import plot_pred_obs
from train_utils import plot_loss_by_fold
from torchgeo.transforms import AugmentationSequential
import kornia.augmentation as K
import torchvision.transforms as T
from custom_datasets import *
import warnings
import logging
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
warnings.filterwarnings("ignore")


def get_RMSE(df):
    
    # Calculate RMSE by fold
    df['SE'] = (df['predicted'] - df['observed']) ** 2
    rmse_by_fold = df.groupby('folds')['SE'].mean() ** 0.5

    # Compute mean RMSE and SD of RMSE values
    mean_rmse = rmse_by_fold.mean()
    sd_rmse = rmse_by_fold.std()

    print(f"RMSE by Fold: {rmse_by_fold}")
    print(f"Mean RMSE = {mean_rmse:.2f} ± {sd_rmse:.2f}\n")
    logging.info(f"RMSE by Fold: {rmse_by_fold}")
    logging.info(f"Mean RMSE = {mean_rmse:.2f} ± {sd_rmse:.2f}\n")

    return mean_rmse, sd_rmse


def train_loop(
        gdf, 
        train_dataloader, 
        model,
        loss_f, 
        optimizer,
        target_variable='C_id',
        ):

    train_loss = 0

    model.train()
    
    for batch in train_dataloader:

        bboxes = batch['bbox']
        imgs_tensor = batch['image']

        if len(imgs_tensor.shape) > 4:
            ## data aug with kornia may create empty dim ?
            imgs_tensor = imgs_tensor.squeeze(1)
        do_forward = True

        imgs_tensor, bboxes, is_non_empty = remove_black_tiles(imgs_tensor, bboxes)

        if is_non_empty == False:
            do_forward = False

        if do_forward:

            labels = list(gdf.loc[gdf['bboxes'].isin(bboxes)][target_variable])
            output = model(imgs_tensor.cuda())
            
            if len(labels)!=output.shape[0]:
                output = correct_duplicate_roi(batch['bbox'], output)
            
            loss = loss_f(output, torch.tensor(labels).cuda())
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_dict = {
            'train_loss' : train_loss,
            }

    return out_dict

def test_loop(
        gdf, 
        test_dataloader, 
        model,
        loss_f, 
        target_variable='C_id',
        ):

    list_predictions = []
    list_gt = []
    list_bbox = []
    valid_loss = 0

    model.eval()

    for batch in test_dataloader:

        bboxes = batch['bbox']
        imgs_tensor = batch['image']

        ## data aug with kornia may create empty dim ?
        if len(imgs_tensor.shape) > 4:
            imgs_tensor = imgs_tensor.squeeze(1)

        do_forward = True
        imgs_tensor, bboxes, is_non_empty = remove_black_tiles(imgs_tensor, bboxes)

        if is_non_empty == False:
            do_forward = False

        if do_forward:

            labels = list(gdf.loc[gdf['bboxes'].isin(bboxes)][target_variable])

            with torch.no_grad():
                preds = model(imgs_tensor.cuda())
                
                if len(labels)!=preds.shape[0]:
                    preds = correct_duplicate_roi(batch['bbox'], preds)
                
                list_predictions += list(preds.detach().cpu().numpy())
                list_gt += labels
                list_bbox += bboxes
                loss = loss_f(preds, torch.tensor(labels).cuda())
                valid_loss += loss.item()

    out_dict = {
            'model' : model,
            'valid_loss' : valid_loss,
            'predictions' : list_predictions,
            'gt' : list_gt,
            'bboxes':list_bbox
            }

    return out_dict


def K_fold(
        K, 
        gdf, 
        dataset, 
        original_model, 
        target_variable='C_id',
        loss_f = nn.CrossEntropyLoss(), 
        bboxes_col_name = 'bboxes', 
        batch_size = 10, 
        n_epochs = 100, 
        image_size=224,
        lr = 10**-5, 
        shuffle_folds=True, 
        out_dir = "",
        fold_seed=42,
        ):

    if not shuffle_folds:
        fold_seed=None # shuffle=False does not work if a seed is passed

    print(f'seed: {fold_seed}')
    logging.info(f'seed: {fold_seed}')

    kf = KFold(n_splits=K, shuffle=shuffle_folds, random_state=fold_seed)

    X = np.array([i for i in range(len(gdf))])

    val_pred = []
    val_gt = []
    val_bboxes = []
    folds = []
    fold_val_losses = {}
    fold_train_losses = {}

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        print(f"Fold {i}:")

        # Create a new instance of the original model
        # otherwise, fold begins with the model at the end of the previous fold
        model = deepcopy(original_model)  

        rois_train = gdf.iloc[train_index][bboxes_col_name]
        rois_test = gdf.iloc[test_index][bboxes_col_name]

        sampler_train = RoiGeoSampler(dataset, size = image_size, rois = rois_train)
        sampler_test = RoiGeoSampler(dataset, size = image_size, rois = rois_test)

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
        best_val_loss = 1_000_000

        optimizer = Adam(model.parameters(), lr=lr)
        # scheduler = ExponentialLR(optimizer, gamma=0.9)
        scheduler = ReduceLROnPlateau(optimizer)
        early_stopper = EarlyStopper(patience=10, min_delta=0.5)

        for j in range(n_epochs):

            print(f"Epoch {j+1}/{n_epochs}", end="\r")

            res_train = train_loop(
                    gdf=gdf, 
                    train_dataloader=train_dataloader, 
                    model=model,
                    optimizer=optimizer,
                    target_variable=target_variable,
                    loss_f=loss_f,
                    )
            res = test_loop(
                    gdf=gdf, 
                    test_dataloader=test_dataloader, 
                    model=model,
                    target_variable=target_variable,
                    loss_f=loss_f,
                    )
            if scheduler:
                scheduler.step(res['valid_loss'])

            training_losses.append(res_train['train_loss'])
            val_losses.append(res['valid_loss'])
            fold_model = res['model']

            if res['valid_loss'] < best_val_loss:
                torch.save(fold_model, os.path.join(out_dir, "best_model_fold_" + str(i) + ".pth"))
                best_val_loss = res['valid_loss']
                best_val_preds = res['predictions']
                best_val_gt = res['gt']
                best_val_bbox = res['bboxes']
            
            if early_stopper.early_stop(res['valid_loss']):             
                break

        fold_val_losses[f'{i}'] = val_losses
        fold_train_losses[f'{i}'] = training_losses
        
        val_pred += best_val_preds
        val_gt += best_val_gt
        val_bboxes += best_val_bbox
        folds += [i] * len(best_val_preds)
        del model

    if str(type(val_pred[0])) == "<class 'numpy.ndarray'>":
        ## may be outputed as a list of one dim np arrays ?
        val_pred = [float(arr[0]) for arr in val_pred]
    res = {
            'predicted' : val_pred,
            'observed' : val_gt,
            'bboxes': val_bboxes,
            'folds': folds,
            }

    losses = {
            'train_losses':fold_train_losses,
            'val_losses':fold_val_losses,
            }

    return res, losses



def train_regression(
        dataset,
        model,
        gdf,
        loss_f,
        bboxes_col_name = 'bboxes',
        target_variable='C_id',
        test_prop = 0.2,
        batch_size=10, 
        n_epochs = 100,
        lr = 10**-5,
        model_save_path = "out/proj.pkl",
        shuffle_folds=False,
        ):
    
    if model_save_path[-4:] == '.pkl':
        len_fn = len(model_save_path.split('/')[-1])
        out_dir = model_save_path[:-len_fn-1]
    else :
        out_dir = model_save_path


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
            n_epochs=n_epochs, 
            lr=lr, 
            shuffle_folds=shuffle_folds, 
            out_dir=out_dir,
            )

    return res, losses

        


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


    # MEANS = [124.19, 87.61, 47.60, 84.60267379987175, 186.40649016850378, -73.23005676269531, 5036.104005604582, 16376.4873046875, 874.5887982478903, -21.899221420288086, -0.04424861477581546, 0.04424861477581546, -16.967143416587476]
    # SDS = [74.40, 53.19, 28.28, 51.12276843481377, 129.5628533539407, 2842.29248046875, 1404.8407341782038, 145635.28125, 162.45557874845173, 2844.227783203125, 0.03822571657935556, 0.03822571657935556, 526.4417166697268]
    MEANS = [124.19, 87.61, 47.60, 0]
    SDS = [74.40, 53.19, 28.28, 1]

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(mean=MEANS, std=SDS),
            K.RandomVerticalFlip(0.5),
            K.RandomHorizontalFlip(0.5),
            K.Resize((patch_h * 14, patch_w * 14)),
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

        train_regression(
                dataset, 
                model, 
                gdf, 
                lr=10**-3,
                test_prop = 0.2, 
                batch_size=10,
                n_epochs=2, 
                loss_f=nn.MSELoss(),
                target_variable=target_variable,
                model_save_path = '../logs/supervised/'
                )
