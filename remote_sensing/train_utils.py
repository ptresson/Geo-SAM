import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from heapq import nsmallest
from sklearn.model_selection import KFold
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix
from utils import remove_black_tiles, correct_duplicate_roi, stack_samples
from samplers import RoiGeoSampler
import hashlib
from copy import deepcopy

def check_model_hash(model):
    model_parameters = model.state_dict()
    model_parameters_bytes = str(model_parameters).encode()
    model_hash = hashlib.sha256(model_parameters_bytes).hexdigest()
    return model_hash


def check_model_dict_hash(model_parameters):
    model_parameters_bytes = str(model_parameters).encode()
    model_hash = hashlib.sha256(model_parameters_bytes).hexdigest()
    return model_hash

def plot_loss_by_fold(losses, cmap='Set1',out_name=None):

    train_losses = losses['train_losses']
    val_losses = losses['val_losses']
    cmap = plt.get_cmap(cmap)
    cmap_iter = iter(cmap.colors)
    colors = [next(cmap_iter) for _ in range(len(train_losses))]

    fig, ax = plt.subplots()
    color_handles = []  # Collect handles for color legend
    style_handles = []  # Collect handles for style legend
    style_handles.append(plt.Line2D([0], [0], color='black', linestyle='-', label=f'Train'))
    style_handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label=f'Validation'))

    for i, (fold, train_fold) in enumerate(train_losses.items()):

        color = colors[i]
        
        val_fold = val_losses[fold]

        train_line, = ax.plot(train_fold, color=color, linestyle='-', label=f'{int(fold)+1}')
        val_line, = ax.plot(val_fold, color=color, linestyle='--',label=f'{int(fold)+1}')
        
        color_handles.append(train_line)  # Add handle for color legend


    # Create color legend
    color_legend = ax.legend(handles=color_handles, title='Folds', loc='upper right', bbox_to_anchor=(0.99, 0.8))
    # Create style legend
    style_legend = ax.legend(handles=style_handles, title='Training/Validation', loc='upper right', bbox_to_anchor=(0.99, 0.99))
    ax.add_artist(color_legend)

    if out_name:
        fig.savefig(out_name)



def do_train_cont(
        train_dataloader, 
        test_dataloader, 
        model,
        loss_f, 
        learning_rate = 10**-5,
        target_variable='C_id',
        ):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    list_y_pred = []
    list_y = []
    train_loss = 0
    valid_loss = 0
    model.train()
    for batch in train_dataloader:
        bboxes = batch['bbox']
        imgs_tensor = batch['image']
        labels = torch.tensor(batch['gt'], dtype = torch.int64)
        if len(imgs_tensor.shape) > 4:
            ## data aug with kornia may create empty dim ?
            imgs_tensor = imgs_tensor.squeeze(1)
        do_forward = True
        imgs_tensor, bboxes, labels, is_non_empty = remove_black_tiles(imgs_tensor, bboxes, labels)
        if is_non_empty == False:
            do_forward = False
        if do_forward:

            imgs_tensor = torch.nan_to_num(imgs_tensor, nan=0)
            output = model(imgs_tensor.cuda())
            list_y_pred += list(output.detach().cpu().numpy())
            list_y += labels

            loss = loss_f(output.type(torch.float32), labels.cuda())
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
    len_train = len(list_y)

    list_y_pred_val = []
    list_y_val = []
    list_bbox = []

    model.eval()

    for batch in test_dataloader:
        bboxes = batch['bbox']
        imgs_tensor = batch['image']
        labels = torch.tensor(batch['gt'], dtype = torch.int64)

        ## data aug with kornia may create empty dim ?
        if len(imgs_tensor.shape) > 4:
            imgs_tensor = imgs_tensor.squeeze(1)
        do_forward = True
        imgs_tensor, bboxes, labels, is_non_empty = remove_black_tiles(imgs_tensor, bboxes, labels)
        if is_non_empty == False:
            do_forward = False

        if do_forward:

            imgs_tensor = torch.nan_to_num(imgs_tensor, nan=0)
            with torch.no_grad():
                preds = model(imgs_tensor.cuda())

                list_y_pred_val += list(preds.detach().cpu().numpy().argmax(1))
                list_y_val += labels
                list_bbox += bboxes
                loss = loss_f(preds.type(torch.float32), labels.cuda())
                valid_loss += loss.item()

    print("Train len : ", len_train)
    print("Val len : ", len(list_y_val))
    train_loss /= len_train
    valid_loss /= len(list_y_val)
    print(f"Avg loss Train: {train_loss:>4f} \t Avg loss Validation: {valid_loss:>4f}")

    # predictions are saved as arrays for some reason ?
    #print([float(arr[0]) for arr in list_y_pred_val] )
    #list_y_pred_val = [float(arr[0]) for arr in list_y_pred_val]       
    print(classification_report(list_y_val, list_y_pred_val, labels=np.unique(list_y_val)))
    print(confusion_matrix(list_y_val, list_y_pred_val, labels=np.unique(list_y_val)))

    out_dict = {
            'model' : model,
            'train_loss' : train_loss,
            'valid_loss' : valid_loss,
            'predictions' : list_y_pred_val,
            'gt' : list_y_val,
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
        N_epochs = 100, 
        image_size=224,
        lr = 10**-5, 
        shuffle_folds = True, 
        out_dir = "",
        fold_seed=42,
        ):

    kf = KFold(n_splits=K, shuffle=shuffle_folds, random_state=fold_seed)

    fold_best_train_losses = np.zeros(K)
    fold_best_val_losses = np.zeros(K)
    X = np.array([i for i in range(len(gdf))])

    val_pred = []
    val_gt = []
    val_bboxes = []
    folds = []
    fold_val_losses = {}
    fold_train_losses = {}

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        print(f"\nFold {i}:")

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
        best_val_loss = 10000

        for j in range(N_epochs):

            print(f"\nEpoch {j+1}")

            res = do_train_cont(
                    train_dataloader, 
                    test_dataloader, 
                    model,
                    loss_f, 
                    learning_rate=lr,
                    target_variable=target_variable,
                    )

            training_losses.append(res['train_loss'])
            val_losses.append(res['valid_loss'])
            fold_model = res['model']

            if res['valid_loss'] < best_val_loss:
                torch.save(fold_model, os.path.join(out_dir, "best_model_fold_" + str(i) + ".pth"))
                best_val_loss = res['valid_loss']
                best_val_preds = res['predictions']
                best_val_gt = res['gt']
                best_val_bbox = res['bboxes']

        fold_val_losses[f'{i}'] = val_losses
        fold_train_losses[f'{i}'] = training_losses
        
        val_pred += best_val_preds
        val_gt += best_val_gt
        val_bboxes += best_val_bbox
        folds += [i] * len(best_val_preds)
        del model

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
