import statistics
from numpy.ma import sort
import torch
from torch.functional import _return_counts
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from collections import OrderedDict
import os
import re
import glob
import timm
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
import sys
import geopandas as gpd
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
from rtree.index import Index, Property
from typing import Any, Dict, List
import warnings
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import rasterio
from PIL import Image
import collections
from heapq import nsmallest
from rasterio.transform import Affine
from shapely.geometry import Polygon
import logging
#from models import SimpleFCHead

class ThreeFCHead(nn.Module):
    def __init__(self, embed_dim, nb_cls, activation='identity'):
        super().__init__()
        self.intermediate_dim = 512
        self.flatten = nn.Flatten()
        if activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'identity':
            activation_func = nn.Identity()
        elif activation == 'softmax':
            activation_func = nn.Softmax()
        else:
            raise ValueError("Invalid activation function")

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, nb_cls),
            activation_func,
        )

    def forward(self, x):
        x = self.flatten(x)
        cls = self.linear_relu_stack(x)
        return cls

class DynamicFCHead(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            nb_cls, 
            num_layers=3, 
            hidden_dim=512, 
            activation='identity'
            ):
        super().__init__()
        self.flatten = nn.Flatten()
        
        if activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'identity':
            activation_func = nn.Identity()
        elif activation == 'softmax':
            activation_func = nn.Softmax()
        else:
            raise ValueError("Invalid activation function")

        layers = []

        if num_layers > 1:  # Add intermediate layers if num_layers is greater than 1
            layers.append(nn.Linear(embed_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 2):  # -2 because we already added the first layer and ReLU
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim if num_layers > 1 else embed_dim, nb_cls))
        layers.append(activation_func)

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        cls = self.linear_stack(x)
        return cls


class FCHead(nn.Module):
    def __init__(self, embed_dim, nb_cls, activation='identity'):
        super().__init__()
        self.flatten = nn.Flatten()
        if activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'identity':
            activation_func = nn.Identity()
        elif activation == 'softmax':
            activation_func = nn.Softmax()
        else:
            raise ValueError("Invalid activation function")

        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, nb_cls),
            activation_func,
        )

    def forward(self, x):
        x = self.flatten(x)
        cls = self.linear_relu_stack(x)
        return cls


def remove_black_tiles(current_rgb_tiles, current_bbox_tiles):
    bs, channels, height, width = current_rgb_tiles.shape
    channel_sum = torch.sum(current_rgb_tiles, dim=1)
    mins = torch.amin(channel_sum, dim=1)
    mins = torch.amin(mins, dim=1)
    indices = mins.nonzero()
    is_non_empty = False
    new_current_bbox_tiles = []
    if len(indices) > 0:
        is_non_empty = True
        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), channels, height, width)
        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])
    return current_rgb_tiles, new_current_bbox_tiles, is_non_empty
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_pretrained_vit(
        model, 
        checkpoint_path ,
        nchannels=1,
        patch_size=14, 
        feat_dim=1024, 
        pos_embed_size=257, 
        ):

    # kernel_size = model.patch_embed.proj.kernel_size
    # stride = model.patch_embed.proj.stride
    # embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # print(model.pos_embed)

    model.patch_embed.proj = nn.Conv2d(nchannels, feat_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    model.pos_embed = nn.Parameter(torch.tensor(model.pos_embed[:, :pos_embed_size]))

    checkpoint = torch.load(checkpoint_path)
    if 'teacher' in checkpoint:
        d = checkpoint['teacher']
        d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
        model.load_state_dict(d2, strict=False)
    if 'model' in checkpoint:
        d = checkpoint['model']
        d2 = OrderedDict([(k, v) for k, v in d.items() if ('decoder_blocks' not in k)])
        model.load_state_dict(d2, strict=False)

    return model

def load_pretrained_vit_and_add_custom_head(
        checkpoint_path=None, 
        model_name='vit_large_patch16_224', 
        patch_size=14, 
        feat_dim=1024, 
        pos_embed_size=257, 
        in_chans=1, 
        out_classes=6, 
        freeze_backbone = True, 
        freeze_proj = False,
        head=None,
        activation='identity'
        ):


    # pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    if checkpoint_path:
        model = timm.create_model(model_name, num_classes=out_classes)
        model.patch_embed.proj = nn.Conv2d(in_chans, feat_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        # model.pos_embed = nn.Parameter(torch.tensor(pretrained.pos_embed[:, :pos_embed_size]))
        checkpoint = torch.load(checkpoint_path)
        try:
            d = checkpoint['teacher']
            d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
            model.load_state_dict(d2, strict=False)
        except:
            d = checkpoint
            d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
            model.load_state_dict(d2, strict=False)

    else :
        model = pretrained
        model= vit_first_layer_with_nchan(model, in_chans)

    if head == None:
        model.head = ThreeFCHead(feat_dim, out_classes, activation)
    else:
        model.head = head

    # print(model)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name[:4]=="head"

    if not freeze_proj:
        for name, param in model.named_parameters():
            if ('patch_embed'in name) and ('proj' in name):
                param.requires_grad = True

    return model


def vit_first_layer_with_nchan(model, in_chans=1):

    kernel_size = model.patch_embed.proj.kernel_size
    stride = model.patch_embed.proj.stride
    embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # copy the original patch_embed.proj config 
    # except the number of input channels
    new_conv = torch.nn.Conv2d(
            in_chans, 
            out_channels=embed_dim,
            kernel_size=kernel_size, 
            stride=stride
            )
    # copy weigths and biases
    weight = model.patch_embed.proj.weight.clone()
    bias = model.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model


def intersects_with_img(roi, file_list):
    res = False
    for file in file_list:
        with rasterio.open(file) as ds :
            tf = ds.meta.copy()['transform']
            bounds = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
            if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
                res = True
                break      
    return res

def intersects_with_torchgeo_dataset(roi, dataset):
    res = False

    bounds = dataset.index.bounds
    print(bounds)

    if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
        res = True

    return res

def get_intersected_bboxes(
        gdf, 
        img_dir, 
        filename_glob, 
        geom_col_name = 'bboxes'
        ):
    pathname = os.path.join(img_dir, "**", filename_glob)
    file_list = []
    for filepath in glob.iglob(pathname, recursive=True):
        file_list.append(filepath)
    return gdf.loc[[intersects_with_img(gdf[geom_col_name][i], file_list) for i in gdf.index]]

def get_x_bbox(bbox):
    try:
        return bbox[0]+bbox[2]
    except:
        return 'n/a'

def correct_duplicate_roi(bboxes_batch, output, labels):
    unique_bboxes = []
    keep_ind = []
    for i in range(len(bboxes_batch)) :
        if bboxes_batch[i] not in unique_bboxes and bboxes_batch[i].maxx-bboxes_batch[i].minx>90 and bboxes_batch[i].maxy-bboxes_batch[i].miny>90:          #TODO: remove hardcoding
            unique_bboxes.append(bboxes_batch[i])
            keep_ind.append(i)
    output = output[keep_ind]
    labels = labels[keep_ind]
    return output, labels

def _list_dict_to_dict_list(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    collated = collections.defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated


def stack_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    collated: Dict[Any, Any] = _list_dict_to_dict_list(samples)
    unique_bboxes = []
    keep_ind = []
    for i in range(len(collated['bbox'])):
        bbox = collated['bbox'][i]
        if bbox not in unique_bboxes and bbox.maxx-bbox.minx>90 and bbox.maxy-bbox.miny>90 :
            unique_bboxes.append(bbox)
            keep_ind.append(i)
    for key, value in collated.items():
        if isinstance(value[0], torch.Tensor):
            if len(value)==1:
                collated[key] = torch.stack(tuple(value))
            else:
                value = np.array(value)[keep_ind]
                collated[key] = torch.stack(tuple(value))
            
    return collated

def prepare_shapefile_dataset(
        shp_path, 
        img_dir, 
        filename_glob, 
        dataset,
        target_variable='C_id',
        geom_col_name = 'bboxes',
        sort_geographicaly=False,
        buffer_size=50,
        ):

    bb = dataset.index.bounds

    def polygon_to_bbox(polygon):
        bounds = list(polygon.bounds)
        bounds[1], bounds[2] = bounds[2], bounds[1]
        return BoundingBox(*bounds, bb[4], bb[5])

    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.dropna(subset=[target_variable])
    if gdf.geom_type.unique() == "Point":
        gdf.geometry = gdf.buffer(buffer_size, cap_style = 3)

    # changes labels id so they go from 0 to N-1, with N the total number of labels. Conserves labels numerical order
    labels = np.array(gdf[target_variable])
    ordered = nsmallest(len(np.unique(labels)), np.unique(labels))
    gdf[target_variable] = [ordered.index(i) for i in labels]

    # only conserves rois which intersect with the images from the dataset
    gdf[geom_col_name] = [polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
    gdf = get_intersected_bboxes(gdf, img_dir, filename_glob)

    if sort_geographicaly:
        gdf['x_temp'] = gdf['bboxes'].apply(get_x_bbox)
        gdf = gdf.sort_values('x_temp')
    gdf = gdf.drop_duplicates()
    gdf.index = [i for i in range(len(gdf))]
    print("Nb roi : ", len(gdf))
    return gdf

def prepare_shapefile_dataset_cont(
        shp_path, 
        img_dir, 
        filename_glob, 
        dataset,
        target_variable='C_id',
        geom_col_name = 'bboxes',
        sort_column=None,
        buffer_size=50,
        normalize=False,
        ):

    bb = dataset.index.bounds

    def polygon_to_bbox(polygon):
        bounds = list(polygon.bounds)
        bounds[1], bounds[2] = bounds[2], bounds[1]
        return BoundingBox(*bounds, bb[4], bb[5])

    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.dropna(subset=[target_variable])
    if gdf.geom_type.unique() == "Point":
        gdf.geometry = gdf.buffer(buffer_size, cap_style = 3)


    # # changes labels id so they go from 0 to N-1, with N the total number of labels. Conserves labels numerical order
    # labels = np.array(gdf[target_variable])
    # ordered = nsmallest(len(np.unique(labels)), np.unique(labels))
    # gdf[target_variable] = [ordered.index(i) for i in labels]

    # only conserves rois which intersect with the images from the dataset
    gdf[geom_col_name] = [polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
    gdf = get_intersected_bboxes(gdf, img_dir, filename_glob)

    if sort_column:
        gdf = gdf.sort_values(sort_column)

    if normalize:
        gdf[target_variable] = (gdf[target_variable] - gdf[target_variable].mean()) / gdf[target_variable].std()

    gdf.index = [i for i in range(len(gdf))]
    print("Nb roi : ", len(gdf))
    return gdf

def resize_image(im_arr, new_size=()):
    if len(new_size)==0:
        return im_arr
    elif len(new_size)==1:
        new_size = new_size*2
    elif len(new_size)>2:
        print("Wrong target dimensions")
        sys.exit()
    image = Image.fromarray(im_arr)
    image = image.resize(new_size)
    return np.array(image)

def crop_duplicate(og_img_path, macro_img, multiple = 224):
    with rasterio.open(og_img_path) as ds :
        shape = ds.read(1).shape
        ds.close()
    print(shape)
    print(macro_img.shape)
    if macro_img.shape[0]>shape[0] and macro_img.shape[1]>shape[1]:
        macro_img = np.delete(macro_img, [multiple+i for i in range(multiple - shape[0]%multiple)], axis=0)
        macro_img = np.delete(macro_img, [multiple*(shape[1]//multiple)+i for i in range(multiple - shape[1]%multiple)], axis=1)
    print(macro_img.shape)
    return macro_img

def array_to_geotiff(
        array, 
        output_file, 
        top_left_corner_coords, 
        pixel_width, 
        pixel_height,
        crs,
        dtype='float32',
        ):
    """
    Convert a numpy array to a GeoTIFF file.
    
    Parameters:
        array (numpy.ndarray): The numpy array representing the raster.
        output_file (str): The path to save the output GeoTIFF file.
        top_left_corner_coords (tuple): Tuple containing the coordinates (x, y) of the top left corner.
        pixel_width (float): Width of a pixel in the raster.
        pixel_height (float): Height of a pixel in the raster.
    """
    from rasterio.transform import from_origin
    # Get the dimensions of the array
    height, width, channels = array.shape
    
    # Define the transformation matrix
    transform = from_origin(top_left_corner_coords[0], top_left_corner_coords[1], pixel_width, pixel_height)
    
    # Create the GeoTIFF file
    with rasterio.open(output_file, 'w', driver='GTiff',
                       height=height, width=width, count=channels, dtype=dtype,
                       crs=crs, transform=transform) as ds:
        ds.write(np.transpose(array, (2, 0, 1)))

def create_bbox_shp(long0, lat0, lat1, long1):
    return Polygon([[long0, lat0], [long1, lat0], [long1, lat1], [long0, lat1]])

def aggregate_overlapping_polygons_with_same_label(gdf, attribute_name, attribute_value):
    polygons = gdf.loc[(gdf[attribute_name] == attribute_value)]
    single_multi_polygon = polygons['geometry'].unary_union
    try:
        polygons = single_multi_polygon.geoms
    except:
        polygons = [single_multi_polygon]
    return polygons

def export_on_map(
        labels, 
        bboxes, 
        crs, 
        out_path='out/test.shp',
        aggregate=False,
        ):  

    fullpathname_cluster_shp = os.path.join(os.getcwd(), out_path)

    bboxes_shp = [create_bbox_shp(bboxes[i][0], 
                                  bboxes[i][2], 
                                  bboxes[i][3], 
                                  bboxes[i][1]
                                  ) for i in range(len(bboxes))]
    labels_shp = labels
    d = {'label': labels_shp, 'geometry': bboxes_shp}
    gdf = gpd.GeoDataFrame(d, crs = crs)
    gdf = gdf[gdf['label'] != -1]

    if aggregate:
        d = {'label': [], 'geometry': []}
        for label in gdf['label'].unique():
            current_polygons = aggregate_overlapping_polygons_with_same_label(gdf, 'label', label)
            for polygon in current_polygons:
                d['label'].append(label)
                d['geometry'].append(polygon)

        gdf = gpd.GeoDataFrame(d, crs = crs)
        print(gdf)
    gdf.to_file(fullpathname_cluster_shp, driver='ESRI Shapefile')


def get_stat_by_band(tif, band_number, stat='STATISTICS_MEAN'):
    '''
    reads metadata of geotif by specifying the band number and desired statistic
    '''
    with rasterio.open(tif) as src:
        statistic = src.tags(band_number)[stat]
    src.close()
    return  statistic


def get_mean_sd_by_band(tif, ignore_zeros=True):
    '''
    reads metadata or computes mean and sd of each band of a geotif
    '''

    src = rasterio.open(tif)
    means = []
    sds = []

    for band in range(1, src.count+1):

        if src.tags(band) != {}: # if metadata are available
            mean = src.tags(band)['STATISTICS_MEAN']
            sd = src.tags(band)['STATISTICS_STDDEV']

        else: # if not, just compute it
            if ignore_zeros:
                arr = src.read(band)
                # mean = np.nanmean(np.where(arr!=0,arr,np.nan),0)
                # sd = np.nanstd(np.where(arr!=0,arr,np.nan),0)
                mean = np.ma.masked_equal(arr, 0).mean()
                sd = np.ma.masked_equal(arr, 0).std()
                del arr # cleanup memory in doubt

            else:    
                arr = src.read(band)
                mean = np.mean(arr)
                sd = np.std(arr)
                del arr # cleanup memory in doubt

        means.append(float(mean))
        sds.append(float(sd))

    src.close()
    return  means, sds

def get_crs(tif):
    with rasterio.open(tif) as src:
        crs = src.crs
    return crs

def remove_empty_tiles(current_rgb_tiles, current_bbox_tiles, ratio=None):

    bs, channels, height, width = current_rgb_tiles.shape
    is_non_empty = False
    new_current_bbox_tiles = []

    # if len(current_rgb_tiles.unique(dim=1)) > channels:
    #     is_non_empty = True
    indices = []
    if ratio :
        ratio = 0.018
        threshold = int(ratio * channels * height * width)
    else:
        threshold = channels

    for i in range(bs):

        print(len(current_rgb_tiles[i].unique()), threshold)
        if len(current_rgb_tiles[i].unique()) > threshold:
            # is_non_empty = True
            indices.append(i)

    if len(indices) > 0:
        is_non_empty = True
        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), channels, height, width)
        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])
    # current_rgb_tiles = current_rgb_tiles[indexes, :, :, :]
    # current_rgb_tiles = current_rgb_tiles.reshape(len(indexes), channels, height, width)
    # for i in indexes:
    #     new_current_bbox_tiles.append(current_bbox_tiles[i])

    return current_rgb_tiles, new_current_bbox_tiles, is_non_empty


def remove_black_tiles2(current_rgb_tiles, current_bbox_tiles, threshold):

    bs, channels, height, width = current_rgb_tiles.shape
    channel_sum = torch.sum(current_rgb_tiles, dim=1)
    # print(channel_sum.shape)
    mins = torch.amin(channel_sum, dim=1)
    # print(mins.shape)
    mins = torch.amin(mins, dim=1)
    # print(mins)
    indices = []

    for i in range(bs):
        if mins[i] > threshold:
            # is_non_empty = True
            indices.append(i)

    is_non_empty = False
    new_current_bbox_tiles = []
    if len(indices) > 0:
        is_non_empty = True
        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), channels, height, width)
        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])
    return current_rgb_tiles, new_current_bbox_tiles, is_non_empty


def remove_black_tiles(current_rgb_tiles, current_bbox_tiles, current_labels):

    # print(current_rgb_tiles.shape)
    bs, channels, height, width = current_rgb_tiles.shape
    channel_sum = torch.sum(current_rgb_tiles, dim=1)
    mins = torch.amin(channel_sum, dim=1)
    mins = torch.amin(mins, dim=1)
    indices = mins.nonzero()
    is_non_empty = False
    new_current_bbox_tiles = []
    new_current_labels = []
    if len(indices) > 0:
        is_non_empty = True
        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), channels, height, width)
        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])
            new_current_labels.append(current_labels[i])
    return current_rgb_tiles, new_current_bbox_tiles, torch.tensor(new_current_labels, dtype = torch.int64), is_non_empty

def remove_black_labels(current_rgb_tiles, current_masks, current_bbox_tiles):
    '''
    Attention ! It is sensible to transforms. 
    Hence, If a rotation fills the mask with zeros, it will be ignored
    '''

    bs, channels, height, width = current_masks.shape
    imgbs, imgchannels, imgheight, imgwidth = current_rgb_tiles.shape
    channel_sum = torch.sum(current_masks, dim=1)
    # print(channel_sum.shape)
    # print(current_masks.shape)
    # print(channel_sum)
    # print(channel_sum.shape)
    mins = torch.amin(channel_sum, dim=1)
    # print(mins.shape)
    # print(mins)
    # print(mins.shape)
    mins = torch.amin(mins, dim=1)
    # print(mins.shape)
    indices = mins.nonzero()
    # print(indices)
    is_non_empty = False
    new_current_bbox_tiles = []
    if len(indices) > 0:
        is_non_empty = True

        current_masks = current_masks[indices, :, :, :]
        current_masks = current_masks.reshape(len(indices), channels, height, width)

        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), imgchannels, imgheight, imgwidth)

        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])
    return current_rgb_tiles, current_masks, new_current_bbox_tiles, is_non_empty

# def remove_black_tiles_mask(current_rgb_tiles, current_masks, current_bbox_tiles):

#     bs, channels, height, width = current_masks.shape
#     imgbs, imgchannels, imgheight, imgwidth = current_rgb_tiles.shape
#     channel_sum = torch.sum(current_rgb_tiles, dim=1)
#     mins = torch.amin(channel_sum, dim=1)
#     mins = torch.amin(mins, dim=1)
#     indices = mins.nonzero()
#     is_non_empty = False
#     new_current_bbox_tiles = []

#     if len(indices) > 0:
#         is_non_empty = True

#         current_masks = current_masks[indices, :, :, :]
#         current_masks = current_masks.reshape(len(indices), channels, height, width)

#         current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
#         current_rgb_tiles = current_rgb_tiles.reshape(len(indices), imgchannels, imgheight, imgwidth)

#         for i in indices:
#             new_current_bbox_tiles.append(current_bbox_tiles[i])

#     return current_rgb_tiles, current_masks, new_current_bbox_tiles, is_non_empty

def remove_black_tiles_mask(current_rgb_tiles, current_masks, current_bbox_tiles):

    bs, channels, height, width = current_masks.shape
    imgbs, imgchannels, imgheight, imgwidth = current_rgb_tiles.shape
    # channel_sum = torch.sum(current_rgb_tiles, dim=1)
    mins = torch.amin(current_rgb_tiles,dim=1)
    # mins = torch.amin(mins, dim=1)
    print(mins)
    # print(len(mins[0]))
    print(len(mins))
    # # uniques = torch.unique(current_rgb_tiles,dim=1)
    # uniques, counts = torch.unique(current_rgb_tiles, dim=0, return_counts=True)
    # print(counts)
    # print(counts.shape)
    sys.exit(1)

    indices = uniques.nonzero()
    is_non_empty = False
    new_current_bbox_tiles = []

    if len(indices) > 0:
        is_non_empty = True

        current_masks = current_masks[indices, :, :, :]
        current_masks = current_masks.reshape(len(indices), channels, height, width)

        current_rgb_tiles = current_rgb_tiles[indices, :, :, :]
        current_rgb_tiles = current_rgb_tiles.reshape(len(indices), imgchannels, imgheight, imgwidth)

        for i in indices:
            new_current_bbox_tiles.append(current_bbox_tiles[i])

    return current_rgb_tiles, current_masks, new_current_bbox_tiles, is_non_empty

def plot_pred_obs(
        predicted, 
        observed, 
        color_variable=None,
        folds=None,
        cmap='Set1',
        label='',
        out_name=None, 
        show=False,
        text_note=None
        ):

        fig, ax = plt.subplots(figsize=(6, 6))
        # Set the aspect ratio to equal for a square plot
        ax.set_aspect('equal', adjustable='box')

        if folds:
            # Assign different colors for each fold
            unique_folds = set(folds)
            # colors = plt.cm.Set1(np.linspace(0, 1, len(unique_folds)))
            cmap = plt.get_cmap(cmap)
            cmap_iter = iter(cmap.colors)
            colors = [next(cmap_iter) for _ in range(len(unique_folds))]

            for fold, color in zip(unique_folds, colors):
                fold_indices = np.where(np.array(folds) == fold)[0]
                ax.scatter(
                    np.array(observed)[fold_indices],
                    np.array(predicted)[fold_indices],
                    label=f'{fold+1}',
                    color=color,
                    alpha=0.7,
                    marker='+'
                )
            ax.legend(title='Fold', bbox_to_anchor=(0.99, 0.01), loc='lower right')

        else:
            # Create a scatter plot of observed vs. predicted values
            if not color_variable:
                color_variable='black'

            scatter = ax.scatter(
                    observed, 
                    predicted, 
                    c=color_variable,
                    cmap=cmap,
                    marker='+', 
                    alpha=0.7,
                    )
            if color_variable != 'black':
                plt.colorbar(scatter, label=label, shrink=0.8)


        # Add a diagonal line representing perfect predictions (x=y)
        amin = min(min(observed), min(predicted))
        amax = max(max(observed), max(predicted))
        x = np.linspace(amin, amax, 100)

        ax.plot(x, x, linestyle='--', c='r')

        ax.set_xlabel('Observed Values')
        ax.set_ylabel('Predicted Values')

        if text_note:
            ax.text(
                    0.1, 
                    0.9, 
                    text_note, 
                    transform=ax.transAxes, 
                    fontsize=12, 
                    va='top'
                    )

        # Show the plot
        if show:
            plt.show()

        # fig.savefig(os.path.join(out_dir, f'pred_obs_{target_variable}.png'))
        if out_name:
            fig.savefig(out_name)

def get_geo_folds(input_shp,
                nfolds=6,
                seed=42,
                output_shp=None,
                # lat_variable='y_utm',
                # long_variable='x_utm'
                ):

    gdf = gpd.read_file(input_shp)
    gdf = gdf[gdf.geometry != None]
    # print(gdf)
    # gdf.reset_index(inplace=True)

    X = []
    for row in gdf.iterrows():
        index, data = row
        X.append([data.geometry.y, data.geometry.x])

    print("===== fitting kmeans =====")
    X = np.array(X)
    kmeans = KMeans(n_clusters=nfolds, random_state=seed).fit(X.astype('double'))
    folds = kmeans.labels_
    # increment all by one to avoir fold at 0
    folds = [x+1 for x in folds]

    # check distribution
    d = {}
    for x in folds:
        d[x] = d.get(x,0) + 1
     
    # printing result
    print(f"The list frequency of elements is : {d}" )

    gdf['geo_fold'] = folds

    if output_shp:
        gdf.to_file(output_shp)

    return gdf

def change_tif_resolution(orig_raster_path, dest_raster_path, new_resolution):

    # Open the original raster
    with rasterio.open(orig_raster_path) as orig_src:
        orig_array = orig_src.read()
        # Get the current resolution
        orig_resolution = orig_src.transform[0]
        # Calculate the factor by which to change the resolution
        resolution_factor = int(orig_resolution / new_resolution)

        # Calculate the new shape of the array
        new_shape = (
            orig_array.shape[0],
            orig_array.shape[1] * resolution_factor,
            orig_array.shape[2] * resolution_factor,
        )

        # Create a new array with the desired resolution
        dest_array = np.zeros(new_shape, dtype=orig_array.dtype)

        # Iterate through the original array and assign values to the new array
        for b in range(orig_array.shape[0]):
            for i in range(orig_array.shape[1]):
                for j in range(orig_array.shape[2]):
                    dest_array[
                            b, 
                            i * resolution_factor : (i + 1) * resolution_factor, 
                            j * resolution_factor : (j + 1) * resolution_factor
                            ] = orig_array[b, i, j]

        # Get the metadata from the original raster
        dest_meta = orig_src.meta.copy()

        # Update metadata with the new resolution
        dest_meta['transform'] = rasterio.Affine(new_resolution, 0, orig_src.transform[2], 0, -new_resolution, orig_src.transform[5])
        dest_meta['width'] = dest_array.shape[2]
        dest_meta['height'] = dest_array.shape[1]

    # Write the new raster
    with rasterio.open(dest_raster_path, 'w', **dest_meta) as dest_dst:
        dest_dst.write(dest_array)

if __name__ == "__main__":
    
    # print(float(get_stat_by_band('/home/ptresson/hegra/crop.tif', 1)))
    # print(get_crs('/home/ptresson/hegra/crop.tif'))
    # means, sds = get_mean_sd_by_band('/home/ptresson/pteryx/pteryx/data/tifs/hegra.tif')
    # print(means)
    # print(sds)
    template = "./out/hegra_raster_cut3.tif"
    arra = np.load('./out/umap-100a.npy')
    arrb = np.load('./out/umap-100b.npy')
    print(arrb.shape)
    print(arra.shape)

    arr = np.zeros((arra.shape[0],arra.shape[1]+arrb.shape[1], arra.shape[2]))
    print(arr.shape)
    # sys.exit(1)

    # arr[0:arra.shape[0],:,:] = arra
    # arr[arra.shape[0]:,:,:] = arrb
    arr[:,:arra.shape[1],:] = arra
    arr[:,arra.shape[1]:,:] = arrb
    np.save("./out/hegra/umap-100.npy", arr)

    # numpy_to_geotif(
    #         template, 
    #         arr,
    #         pixel_scale=100, 
    #         dtype='float32', 
    #         out_path='./out/hegra/umap-100.tif'
    #         )




