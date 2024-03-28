import torch
import numpy as np
from PIL import Image
import sys
import rasterio
import os
import geopandas as gpd
import glob
from torchgeo.datasets.utils import BoundingBox
import json
import matplotlib.pyplot as plt
from heapq import nsmallest

def vit_first_layer_with_nchan(model, in_chans=1, embed_dim=1024, patch_size=14,):
            new_conv = torch.nn.Conv2d(in_chans, embed_dim,
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
            weight = model.patch_embed.proj.weight.clone().cuda()
            bias = model.patch_embed.proj.bias.clone().cuda()
            with torch.no_grad():
                for i in range(0,in_chans):
                    j = i%3 # cycle every 3 bands
                    new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
                    new_conv.bias[:] = bias[:]
                model.patch_embed.proj = new_conv

                return model
            

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


def reconstruct_img(div_images, Nx, Ny):
    #assert Nx*Ny == len(div_images)
    agreg_X = div_images[0]
    for i in range(1, Nx):
        agreg_X = np.concatenate((agreg_X, div_images[i]), axis=1)
        agreg_X = np.asarray(agreg_X, dtype = np.float16)
    agreg = agreg_X
    for j in range(1, Ny):
        agreg_X = div_images[j*Nx]
        for i in range(1, Nx):
            if j*Nx + i < len(div_images):
                agreg_X = np.concatenate((agreg_X, div_images[j*Nx + i]), axis=1)
            else :
                agreg_X = np.concatenate((agreg_X, np.zeros(div_images[0].shape)), axis=1)
            agreg_X = np.asarray(agreg_X, dtype = np.float16)
        agreg = np.concatenate((agreg_X, agreg), axis=0)
    return agreg


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

def polygon_to_bbox(polygon):
    bounds = list(polygon.bounds)
    bounds[1], bounds[2] = bounds[2], bounds[1]
    return BoundingBox(*bounds, 0.0, 9.223372036854776e+18)
    
def get_intersected_bboxes(gdf, img_dir, filename_glob, geom_col_name = 'bboxes'):

    def is_inside_img(roi, file_list):
        res = False
        for file in file_list:
            with rasterio.open(file) as ds :
                tf = ds.meta.copy()['transform']
                bounds = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
                if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
                    res = True
                    break      
        return res
    
    pathname = os.path.join(img_dir, "**", filename_glob)
    file_list = []
    for filepath in glob.iglob(pathname, recursive=True):
        file_list.append(filepath)
    return gdf.loc[[is_inside_img(gdf[geom_col_name][i], file_list) for i in gdf.index]]


def prepare_shapefile_dataset(shp_path, img_dir, filename_glob, geom_col_name = 'bboxes'):
    """
    Returns a geodataframe with all the rois 
    """

    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    
    #if we want labels to range from 0 to Nlabels
    """
    labels = np.array(gdf['C_id'])
    ordered = nsmallest(len(np.unique(labels)), np.unique(labels))
    gdf['C_id'] = [ordered.index(i) for i in labels]
    """
    gdf['bboxes'] = [polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
    gdf = get_intersected_bboxes(gdf, img_dir, filename_glob, geom_col_name)
    gdf = gdf.drop_duplicates()
    gdf.index = [i for i in range(len(gdf))]
    print("Nb roi : ", len(gdf))
    return gdf



def show_loss(train_metrics_path):

    def correct_json_format(json_path):    #for dinov2 training metrics json
        with open(json_path, 'r+') as file : 
            a = file.read()
            splitted = a.split('}')
            updated_str = splitted[0]
            for i in range(1, len(splitted)-1):
                updated_str += '},' + splitted[i]
            updated_str = '[' + updated_str + '}]'
            file.truncate(0)
            
            file.close()
        file = open(json_path, 'w+')
        file.write(updated_str)
        file.close()

    try:
        data = json.load(open(train_metrics_path))
    except :
        correct_json_format(train_metrics_path)
        data = json.load(open(train_metrics_path))
    N = len(data)
    losses = ['total_loss', 'dino_local_crops_loss', 'dino_global_crops_loss', 'ibot_loss']             #koleo loss

    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    X = np.array([i for i in range(N)])
    for loss in losses :
        Y = np.zeros(N)
        for j in range(N):
            Y[j] = data[j][loss]             
        plt.plot(X, Y, label = loss)
    plt.legend()
    plt.show()


def attention_stats_shp(shp_path, attention_tif_path):
    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    
    with rasterio.open(attention_tif_path) as ds :
        tf = ds.meta.copy()['transform']
        bb = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
        gdf = gdf.loc[((gdf['geometry'].bounds['maxx']>bb[0]) &(gdf['geometry'].bounds['maxy']>bb[2]) &(gdf['geometry'].bounds['minx']<bb[1]) &(gdf['geometry'].bounds['miny']<bb[3]))]
        gdf.index = [i for i in range(len(gdf))]
        gdf['bboxes'] = [polygon_to_bbox(gdf['geometry'][i]) for i in range(len(gdf))]
        im = np.transpose(ds.read(), (1,2,0))
        print("image shape : ", im.shape)
        nb_chan = im.shape[-1]
        mean = []
        std = []
        for roi in gdf['bboxes']:
            bottom_left = ds.index(roi[0], roi[3])
            top_right = ds.index(roi[1], roi[2])
            roi_attention = im[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]
            mean.append(np.array([np.mean(roi_attention[:,:,k]) for k in range(nb_chan)]))
            std.append(np.array([np.std(roi_attention[:,:,k]) for k in range(nb_chan)]))

        gdf['attention_mean'] = mean
        gdf['attention_std'] = std
            
    return gdf, np.array(mean), np.array(std)