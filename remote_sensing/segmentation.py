# cf. https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DINOv2

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
from kornia.enhance.normalize import Normalize
import torch.utils.tensorboard as tensorboard
# import timm

# torchgeo
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
from torchgeo.transforms import AugmentationSequential
from rasterio.crs import CRS
import numpy as np

from samplers import RoiGeoSampler, RandomOutsideRoiGeoSampler
from samplers import GridShpGeoSampler, GridOutsideGeoSampler
from utils import intersects_with_img, get_intersected_bboxes, prepare_shapefile_dataset, remove_black_labels
from utils import vit_first_layer_with_nchan, reconstruct_img, resize_image, crop_duplicate, load_pretrained_vit
from utils import numpy_to_geotif
from utils import export_on_map
from utils import get_mean_sd_by_band, get_crs
from utils import remove_black_tiles as remove_black_tiles_fn
from utils import remove_black_tiles2
from utils import remove_black_tiles_mask
from utils import remove_empty_tiles

from torchmetrics import JaccardIndex, Accuracy
import os
from typing import Optional
import tempfile
import sys
import time
import resource
import matplotlib.pyplot as plt


# # pip install -q git+https://github.com/huggingface/transformers.git datasets
# from transformers import Dinov2Model, Dinov2PreTrainedModel
# from transformers.modeling_outputs import SemanticSegmenterOutput
# import evaluate

class Raster(RasterDataset):
    # filename_glob = 'crop_pilote.tif'
    # filename_glob = 'hegra.tif'
    filename_glob = 'hegra_raster_cut3.tif'
    is_image = True

class RGBNIR(RasterDataset):
    # filename_glob = 'crop_pilote.tif'
    # filename_glob = 'hegra.tif'
    filename_glob = 'hegra_raster_cut3.tif'
    is_image = True
    all_bands=[0,1,2,3] # take only RGBNIR wich are the firts 4 bands 


class MNT(RasterDataset):
    filename_glob = 'hegra_mnt_cut.tif'
    is_image = True

class MNS(RasterDataset):
    filename_glob = 'hegra_mns_cut.tif'
    is_image = True

class Map(RasterDataset):
    # filename_glob = 'rasterized_vector.tif' # rasterized version of habitat 
    filename_glob = 'hegra_mask_cut.tif' # rasterized version of habitat 
    is_image = False # so its understood as a mask

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)
        # TODO add relu ?

        return self.classifier(embeddings)

class DINOv2Segmentation(torch.nn.Module):
    def __init__(self, backbone, nclasses, img_size=224):
        super(DINOv2Segmentation, self).__init__()

        self.dinov2 = backbone
        kernel_size = backbone.patch_embed.proj.kernel_size
        embed_dim = backbone.patch_embed.proj.out_channels # corresponds to embed_dim

        self.classifier = LinearClassifier(embed_dim, int(img_size/kernel_size[0]), int(img_size/kernel_size[0]), nclasses)

    def forward(self, images):
        
        # To use with original DINOv2 backbone and not HuggingFace implementation
        feature_dict = self.dinov2.forward_features(images)
        patch_embeddings = feature_dict['x_norm_patchtokens']

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=images.shape[2:], mode="bilinear", align_corners=False)

        return logits


def vit_first_layer_with_nchanhf(model, in_chans=1):
    kernel_size = model.dinov2.embeddings.patch_embeddings.projection.kernel_size
    stride = model.dinov2.embeddings.patch_embeddings.projection.stride
    embed_dim = model.dinov2.embeddings.patch_embeddings.projection.out_channels
    # copy the original patch_embed.proj config 
    # except the number of input channels
    new_conv = torch.nn.Conv2d(
            in_chans, 
            out_channels=embed_dim,
            kernel_size=kernel_size, 
            stride=stride
            )
    # copy weigths and biases
    weight = model.dinov2.embeddings.patch_embeddings.projection.weight.clone()
    bias = model.dinov2.embeddings.patch_embeddings.projection.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.dinov2.embeddings.patch_embeddings.projection = new_conv

    return model


def train_one_epoch(
        model, 
        dataloader, 
        loss_fn, 
        remove_black_tiles=True
        ):

    nsamples = 0
    for idx, batch in enumerate(dataloader):
        print(f"{idx/len(dataloader):.2%}", end="\r")
        batch_bboxes = batch['bbox']
        images = batch["image"].to(device)
        images = images.squeeze(1) # transformation creates fake channel ?
        labels = batch["mask"].long().to(device)

        do_forward = True

        if remove_black_tiles:
            images, labels, batch_bboxes, is_non_empty = remove_black_labels(images, labels, batch_bboxes)
            if is_non_empty == False:
                do_forward = False
      
        if do_forward:

            nsamples += images.shape[0]

            with torch.cuda.amp.autocast():
                # forward pass
                logits = model(images)
                loss = loss_fn(logits, labels.squeeze(1)) # otherwise does not work with batch_size = 1 ??

                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

    print(f"\nTrained on {nsamples} samples")

    return loss.item()

def test_one_epoch(
        model, 
        dataloader,
        loss_fn, 
        remove_black_tiles=True
        ):

    glob_iou = 0.0
    glob_acc = 0.0
    # by removing black tiles, the number of batches may be different from len(dataloader)
    nbatches = 0 

    for idx, batch in enumerate(dataloader):

        print(f"{idx/len(dataloader):.2%}", end="\r")
        batch_bboxes = batch['bbox']
        images = batch["image"].to(device)
        images = images.squeeze(1) # transformation creates fake channel ?
        labels = batch["mask"].long().to(device)
        do_forward = True

        if remove_black_tiles:
            images, labels, batch_bboxes, is_non_empty = remove_black_labels(images, labels, batch_bboxes)
            if is_non_empty == False:
                do_forward = False

        if do_forward:

            nbatches += 1

            with torch.no_grad():
              # forward pass
              logits = model(images)
              loss = loss_fn(logits, labels.squeeze(1)) # otherwise does not work with batch_size = 1 ??
              predicted = logits.argmax(dim=1)
              acc = (predicted == labels.squeeze(1)).float().mean()

              # note that the metric expects predictions + labels as numpy arrays
              miou = jaccard(predicted.detach().cpu(), labels.squeeze(1).detach().cpu())
              acc = acc.detach().cpu()
              glob_iou += miou
              glob_acc += acc

    final_miou = glob_iou /nbatches
    final_acc = glob_acc /nbatches

    return final_miou, final_acc, loss.item()

def mem_usage():
    memory_usage_rss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    memory_usage_rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss /1024
    return memory_usage_rss_self + memory_usage_rss_children


def init_model(
        id2label, 
        mode="HF", 
        checkpoint_path=None, 
        im_size=224,
        in_chans=6,
        ):

    # if mode=="HF":
    #     model = Dinov2ForSemanticSegmentation.from_pretrained(
    #             "facebook/dinov2-base", 
    #             id2label=id2label, 
    #             num_labels=len(id2label),
    #             ignore_mismatched_sizes=True,
    #             )
    #     print(model)
    #     model = vit_first_layer_with_nchanhf(model, in_chans=6)

    if mode=='dinov2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        backbone = vit_first_layer_with_nchan(backbone, in_chans=in_chans)
        model = DINOv2Segmentation(backbone, len(id2label),im_size)

    # if mode=="timm":
    #     backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
    #     backbone = load_pretrained_vit(
    #             backbone,
    #             checkpoint_path, 
    #             nchannels=4,
    #             pos_embed_size=257,
    #             feat_dim=768,
    #             patch_size=16,
    #             )
    #     backbone.reset_classifier(0,'avg')
    #     backbone = vit_first_layer_with_nchan(backbone, in_chans=6)
    #     model = Dinov2ForSemanticSegmentation_timm(backbone, len(id2label))

    return model

def show_sample_prediction(
        dataloader, 
        model, 
        epoch, 
        writer, 
        remove_black_tiles=True,
        verbose=False
        ):

    for batch in dataloader:
        batch_bboxes = batch['bbox']
        images = batch["image"]
        image = images.squeeze(1)
        mask = batch["mask"].long()
        do_forward = True

        if remove_black_tiles:
            image, mask, batch_bboxes, is_non_empty = remove_black_labels(image, mask, batch_bboxes)
            if is_non_empty == False:
                if len(mask.unique()) == 1:
                    do_forward = False
      
        if do_forward:

            with torch.no_grad():
              # forward pass
              logits = model(image.cuda())
              predicted = logits.argmax(dim=1)
              predicted.detach().cpu()

              if verbose:
                  print('mask')
                  print(f'min: {mask.min()}')
                  print(f'max : {mask.max()}')
                  print(f'uniques: {len(mask.unique())}')
                  print('predicted')
                  print(f'min: {predicted.min()}')
                  print(f'max : {predicted.max()}')
                  print(f'uniques: {len(predicted.unique())}')
              image = image[0]
              mask = mask[0]
              predicted = predicted[0]
              predicted = predicted.unsqueeze(0)

              # image = image[:3,:,:] # only keep the last 3 bands
              # image = (image - image.min()) / (image.max() - image.min()) * 255
              image = image[:3,:,:]
              image = (image - image.min()) / (image.max() - image.min())
              mask = (mask - mask.min()) / (mask.max() - mask.min()) * 225
              predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min()) * 225
              # image = image.detach().numpy()
              # image = image.astype(np.uint8)
              mask = mask.detach().numpy()
              mask = mask.astype(np.uint8)
              predicted = predicted.detach().cpu().numpy()
              predicted = predicted.astype(np.uint8)

              writer.add_image('predictions/test image', image, epoch)
              writer.add_image('predictions/test mask', mask, epoch)
              writer.add_image('predictions/predicted', predicted, epoch)

              break


def dataloader_transform_checkup(
        dataloader, 
        writer=None, 
        remove_black_tiles=True,
        save_to_file=False,
        ):

    for batch in dataloader:
        batch_bboxes = batch['bbox']
        images = batch["image"]
        image = images.squeeze(1)
        mask = batch["mask"].long()
        do_forward = True

        if remove_black_tiles:
            image, mask, batch_bboxes, is_non_empty = remove_black_labels(image, mask, batch_bboxes)
            if is_non_empty == False:
                if len(mask.unique()) == 1:
                    do_forward = False
      
        if do_forward:

            ## Checkup if input and output format are as expected
            image = image[0]
            # image = image[0][5]
            # mns = image[4]
            # mnt = image[5]
            # print(mns.mean())
            # print(mns.min())
            # print(mnt.mean())
            # print(mnt.min())
            print('image:')
            print(image.shape)
            # sys.exit(1)
            print(f'min : {image.min()}')
            print(f'max : {image.max()}')
            print(f'uniques : {len(image.unique())}')

            mask = mask[0]
            print('mask:')
            print(mask.shape)
            print(f'min: {mask.min()}')
            print(f'max : {mask.max()}')
            print(f'uniques: {len(mask.unique())}')

            image = image[:3,:,:]
            image = (image - image.min()) / (image.max() - image.min())
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 225
            if writer:
                writer.add_image('train image/image', image)
                writer.add_image('train image/mask', mask)

            if save_to_file:

                ## convert to numpy compliant HWC 
                image = image.permute(1,2,0)
                mask = mask.permute(1,2,0)

                plt.imshow(image)
                plt.savefig('out/image.png')
                plt.imshow(mask)
                plt.savefig('out/mask.png')
            break

def init_chesapeak_dataset():

    from torchgeo.datasets import NAIP, ChesapeakeDE
    from torchgeo.datasets.utils import download_url

    naip_root = os.path.join(tempfile.gettempdir(), "naip")
    naip_url = (
            "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
            )
    tiles = [
            "m_3807511_ne_18_060_20181104.tif",
            "m_3807511_se_18_060_20181104.tif",
            "m_3807512_nw_18_060_20180815.tif",
            "m_3807512_sw_18_060_20180815.tif",
            ]
    for tile in tiles:
        download_url(naip_url + tile, naip_root)

    naip = NAIP(naip_root)

    chesapeake_root = os.path.join(tempfile.gettempdir(), "chesapeake")
    chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)

    dataset = naip & chesapeake

    return dataset


def check_sampling(
        dataloader, 
        crs,
        remove_black_tiles=True, 
        out_path='out/sampling.shp'
        ):

    all_bboxes = []

    for idx, batch in enumerate(dataloader):
        # print(f"{idx/len(dataloader):.2%}", end="\r")
        batch_bboxes = batch['bbox']
        images = batch["image"]
        images = images.squeeze(1) # transformation creates fake channel ?
        labels = batch["mask"].long()

        do_forward = True

        if remove_black_tiles:
            # images, labels, batch_bboxes, is_non_empty = remove_black_labels(images, labels, batch_bboxes)
            # images, batch_bboxes, is_non_empty = remove_empty_tiles(images, batch_bboxes)
            images, batch_bboxes, is_non_empty = remove_black_tiles2(images, batch_bboxes, -77.04)
            # images, new_batch_bboxes, is_non_empty = remove_black_tiles_fn(images, batch_bboxes)
            # images, labels, new_batch_bboxes, is_non_empty = remove_black_labels(images, labels, batch_bboxes)
            # print(len(new_batch_bboxes))
            # print(images.shape)
            if is_non_empty == False:
                print('pouet')
                do_forward = False

        if do_forward:
            all_bboxes.extend(batch_bboxes)

            # # get mode of GT
            # vals,counts = np.unique(mask, return_counts=True)
            # index = np.argmax(counts)
            # labels.extend(vals[index])
    
    shp_labels = [0] * len(all_bboxes)
    print(len(all_bboxes))
    
    export_on_map(
        shp_labels,
        all_bboxes,
        crs,
        out_path=out_path
        )

    return

def inference(
        dataset,
        model,
        batch_size = 10, 
        size=224, # size of sampled images
        roi=None, # defaults takes all dataset bounds
        path_out=None,
        ):

    # if roi is samller than full dataset
    if roi:
        sampler = GridGeoSampler(dataset, size = size, stride=size, roi = roi)
    else:
        sampler = GridGeoSampler(dataset, size = size, stride=size)

    dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            collate_fn=stack_samples, 
            shuffle=False, 
            batch_size = batch_size
            )

    N = len(dataloader)

    bboxes = []

    # initialize tensor used to fit projection/cluster at the end.
    # if working at patch level, we have more individual samples in the end
    if cls_token:
        feat_img = np.zeros((N*batch_size, out_dim))
    else :
        feat_img = np.zeros((N*batch_size,patch_h,patch_w, out_dim))

    i=0
    for batch in dataloader:
        print(f"{i/N:.2%}", end="\r")

        images = batch['image']
        images = images.type(torch.float)
        images = images.cuda()

        for sample in unbind_samples(batch):
            bboxes.append(sample['bbox'])

        with torch.no_grad():

            features_dict = model.forward_features(images)
            if cls_token:
                feat = features_dict['x_norm_clstoken']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(batch_size, feat_dim)
                red_features = embedder.transform(feat)
                red_features = red_features.reshape(batch_size, out_dim)
                feat_img[batch_size*i : batch_size*(i+1)] = red_features
            else:
                feat = features_dict['x_norm_patchtokens']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(batch_size*patch_h*patch_w, feat_dim)
                red_features = embedder.transform(feat)
                red_features = red_features.reshape(batch_size, patch_h, patch_w, out_dim)
                # feat_img[batch_size*i*patch_h*patch_w : batch_size*(i+1)*patch_h*patch_w] = feat
                feat_img[batch_size*i : batch_size*(i+1)] = red_features
        i+=1

    Nx = len(bboxes)
    for i in range(1, len(bboxes)):
        if bboxes[i][0]<bboxes[i-1][0]:
            Nx = i
            break
    Ny = int(len(bboxes)/Nx)
    # print("Nx : ", Nx, "\tNy : ", Ny)

    if cls_token:
        macro_img = (reconstruct_img(feat_img, Nx, Ny))
    
    else:
        macro_img = (reconstruct_img(feat_img, Nx, Ny)*255).astype(np.uint8)
        macro_img = crop_duplicate(os.path.join(img_dir, img_glob), macro_img, multiple=14)

    if path_out:
        np.save(path_out, macro_img)
    return macro_img, bboxes

if __name__ == "__main__":


    ## init tensorboard
    out_dir = 'out/runs/'
    writer = tensorboard.SummaryWriter(out_dir)

    ## sampling size on the raster in pixels
    ## /!\ different from model imput size
    size = 224
    # size = 1000

    ## number of random samples during training
    nsamples = 5_000
    # nsamples = 500

    ## remove black tiles
    remove_black_tiles = False
    # remove_black_tiles = True

    ## ignore a class as BG ?
    ignore_index = 0
    # ignore_index = None

    # ######################### ChesapeakeDE ############################

    # ## correct number of classes for ChesapeakeDE
    # id2label = {
    #     0: (0, 0, 0, 0),
    #     1: (0, 197, 255, 255),
    #     2: (0, 168, 132, 255),
    #     3: (38, 115, 0, 255),
    #     4: (76, 230, 0, 255),
    #     5: (163, 255, 115, 255),
    #     6: (255, 170, 0, 255),
    #     7: (255, 0, 0, 255),
    #     8: (156, 156, 156, 255),
    #     9: (0, 0, 0, 255),
    #     10: (115, 115, 0, 255),
    #     11: (230, 230, 0, 255),
    #     12: (255, 255, 115, 255),
    #     13: (197, 0, 255, 255),
    # }


    # # ## Imagenet means and sd adapted for ChesapeakeDE
    # MEANS = [0.485*255, 0.456*255, 0.406*255, 0.485*255]
    # SDS = [0.229*255, 0.224*255, 0.225*255,0.229*255]

    # train_dataset = init_chesapeak_dataset()
    # val_dataset = init_chesapeak_dataset()

    # # in_chans = 4



    ########################## Hegra ############################

    # MEANS = [3211.0983403106256, 2473.9249186805423, 1675.3644890560977, 4335.811473646549, 3858.9351549515964, 1335.5861093284207] 
    # SDS = [1222.8046380984822, 811.0329354546556, 511.7795012804498, 976.456252068188, 1056.5672047424875, 354.43969977035744] 
    MEANS = [3211.0983403106256, 2473.9249186805423, 1675.3644890560977, 4335.811473646549,789.30445481992, 788.94334968111] 
    SDS = [1222.8046380984822, 811.0329354546556, 511.7795012804498, 976.456252068188,24.758130975788,24.832657292941] 

    ZEROS = [0.] * len(MEANS)
    ZEROS = [a-b for a,b in zip(ZEROS,MEANS)]
    ZEROS = [a/b for a,b in zip(ZEROS,SDS)]
    print(ZEROS)
    ZERO_SUMS = sum(ZEROS)
    print(ZERO_SUMS)

    # zero values for each band after normalize
    # ZEROS = [-2.6260109262457756, -3.0503384147940786, -3.273606083995958, -4.440354050130829, -3.6523328924373697, -3.768161721708237]

    # raster = Raster('./out/') #change to crop_pilote_mask if not working ?
    image = RGBNIR('./out/') #change to crop_pilote_mask if not working ?
    mns = MNS('./out') #change to crop_pilote_mask if not working ?
    mnt = MNT('./out/') #change to crop_pilote_mask if not working ?
    raster = image & mns
    raster = raster & mnt
    map = Map('./out/',) #change to crop_pilote_mask if not working ?

    # names from tuto dataset but correct number of classes for Hegra
    id2label = {
        0: "background",
        1: "candy",
        2: "egg tart",
        3: "french fries",
        4: "chocolate",
        5: "biscuit",
        6: "popcorn",
        7: "pudding",
        8: "ice cream",
        9: "cheese butter",
    }

    train_dataset = raster & map
    val_dataset = raster & map

    ############################# SNR ###########################

    # raster = Raster('/home/ptresson/sharaan/', transforms=rastertransform) #change to crop_pilote_mask if not working ?
    # raster = Raster('/home/ptresson/sharaan/') #change to crop_pilote_mask if not working ?
    # map = Map('./out/',) #change to crop_pilote_mask if not working ?

    # # names from tuto dataset but correct number of classes for SNR
    # id2label = {
    #     0: "background",
    #     1: "candy",
    #     2: "egg tart",
    #     3: "french fries",
    #     4: "chocolate",
    #     5: "biscuit",
    #     6: "popcorn",
    #     7: "pudding",
    #     8: "ice cream",
    #     9: "cheese butter",
    #     10: "cake",
    # }



    ########################## Transforms ######################

    train_transforms = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomShear(10,p=0.1),
            K.RandomAffine(degrees=50.0, p=0.1),
            K.RandomPerspective(p=0.1, distortion_scale=0.4),
            K.RandomHue((0.1,0.1),p=0.1),
            K.ColorJiggle(0.1,0.1,0.1,p=0.1),
            K.RandomRotation(degrees=180, p=0.1),
            K.Resize((224, 224)), # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image", "mask"],
            )

    val_transforms = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image", "mask"],
            )

    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms



    bb = train_dataset.index.bounds

    print(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5])
    med_x = (bb[0] + bb[1])/2
    med_y = (bb[2] + bb[3])/2
    third_x = bb[0] + (bb[1]-bb[0])*1/3
    third_y = bb[2] + (bb[3]-bb[2])*1/3
    twothird_x = bb[0] + (bb[1]-bb[0])*2/3
    twothird_y = bb[2] + (bb[3]-bb[2])*2/3
    quarter_x = bb[0] + (bb[1]-bb[0])*1/4
    quarter_y = bb[2] + (bb[3]-bb[2])*1/4
    threequarter_x = bb[0] + (bb[1]-bb[0])*3/4
    threequarter_y = bb[2] + (bb[3]-bb[2])*3/4
    fifth_x = bb[0] + (bb[1]-bb[0])*1/5
    fifth_y = bb[2] + (bb[3]-bb[2])*1/5
    fourfifth_x = bb[0] + (bb[1]-bb[0])*4/5
    fourfifth_y = bb[2] + (bb[3]-bb[2])*4/5
    sixth_y = bb[2] + (bb[3]-bb[2])*1/6

    train_roi = BoundingBox(fifth_x, twothird_x, sixth_y, threequarter_y, bb[4], bb[5])
    # train_roi = BoundingBox(med_x, bb[1], bb[2], bb[3], bb[4], bb[5])
    # test_roi = [BoundingBox(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5])]
    # test_roi = BoundingBox(bb[0], med_x, bb[2], bb[3], bb[4], bb[5])

    # train_sampler = GridOutsideGeoSampler(train_dataset,size,size, rois=train_roi)
    # train_sampler = GridShpGeoSampler(train_dataset,size,size, rois=train_roi)
    train_sampler = RandomGeoSampler(train_dataset,size,nsamples, roi=train_roi)
    # val_sampler = RandomOutsideRoiGeoSampler(train_dataset,size,nsamples, rois=test_roi)
    val_sampler = GridOutsideGeoSampler(val_dataset,size, int(size * 1.5), rois=[train_roi]) # every two to take less time
    # val_sampler = GridGeoSampler(val_dataset,size, int(size * 1.5), roi=test_roi) # every two to take less time

    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=8, 
            sampler=train_sampler, 
            shuffle=False, 
            collate_fn=stack_samples,
            # num_workers=3,
            )

    val_dataloader = DataLoader(
            val_dataset,  # update if working
            batch_size=8, 
            sampler=val_sampler, 
            shuffle=False, 
            collate_fn=stack_samples,
            )

    # check_sampling(
    #         train_dataloader, 
    #         train_dataset.crs,
    #         remove_black_tiles=False,
    #         out_path='out/train_sampling.shp'
    #         )

    # check_sampling(
    #         val_dataloader, 
    #         val_dataset.crs,
    #         remove_black_tiles=False,
    #         out_path='out/val_sampling.shp'
    #         )

    check_sampling(
            val_dataloader, 
            val_dataset.crs,
            remove_black_tiles=True,
            # remove_black_tiles=False,
            out_path='out/sampling.shp'
            )
    sys.exit(1)

    dataloader_transform_checkup(
            train_dataloader, 
            writer, 
            remove_black_tiles=True,
            )


    model = init_model(
            id2label, 
            mode='dinov2',
            in_chans=len(MEANS),
            # checkpoint_path='./mae_sharaan.pth',
            # im_size=size,
            )

    for name, param in model.named_parameters():
      if name.startswith("dinov2"):
        param.requires_grad = False

    for name, param in model.named_parameters():
      if name.startswith("dinov2.patch_embed"):
        print(name)
        param.requires_grad = True

    jaccard = JaccardIndex(
            task='multiclass', 
            num_classes=len(id2label), 
            ignore_index=ignore_index, # get mIoU ignoring bg
            )

    accuracy = Accuracy(
            task='multiclass',
            num_classes=len(id2label), 
            ignore_index=ignore_index, # get mIoU ignoring bg
            )

    # training hyperparameters
    # NOTE: I've just put some random ones here, not optimized at all
    # feel free to experiment, see also DINOv2 paper
    learning_rate = 5e-5
    epochs = 200

    if ignore_index:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) # if bg is ignored
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # put model on GPU (set runtime to GPU in Google Colab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # put model in training mode
    model.train()
    best_miou = 0

    for epoch in range(epochs):
      print("Epoch:", epoch)

      loss = train_one_epoch(
              model, 
              train_dataloader, 
              loss_fn, 
              remove_black_tiles=False,
              )

      writer.add_scalar('loss/train loss', loss, epoch) # loging with tensorboard

      print("TEST    ")

      miou, acc, test_loss = test_one_epoch(
              model, 
              val_dataloader, 
              loss_fn,
              remove_black_tiles=True,
              )

      print("Mean_iou:", miou)
      print("Accuracy:", acc)

      writer.add_scalar('metric/miou', miou, epoch) # loging with tensorboard
      writer.add_scalar('metric/accuracy', acc, epoch) # loging with tensorboard
      writer.add_scalar('loss/test loss', test_loss, epoch) # loging with tensorboard

      ## export sample prediction to tensorboard to check how its going
      show_sample_prediction(
              val_dataloader,
              model,
              epoch,
              writer,
              remove_black_tiles=True
              )
      if miou > best_miou:
        torch.save(model.state_dict(),f"{out_dir}best.pth")
        best_miou = miou





