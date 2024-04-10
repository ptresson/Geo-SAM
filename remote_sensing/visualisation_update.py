import sys
import warnings

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import timm

import rasterio
import geopandas as gpd

# torchgeo
from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.transforms import AugmentationSequential

# data
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import KMeans
# import hdbscan
warnings.filterwarnings("ignore")

# custom modules
from utils.geo import change_tif_resolution
from utils.geo import array_to_geotiff


def get_features(
        model:nn.Module, 
        images:torch.Tensor, 
        cls_token=False,
        ):
    """
    A wrapper function to get module form diffrent ViT implementation. Will probably be obsolete with newer timm versions
    """

    if cls_token:
        if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
            features = model.forward_features(images)
            if isinstance(features, dict): # DINOv2 implementation
                return features['x_norm_clstoken']
            else: # timm implementation
                return features[:,0,:] 
        else:
            return model(images)
    else:
        if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
            features = model.forward_features(images)
            if isinstance(features, dict):
                return features['x_norm_patchtokens']
            else:
                return features[:,1:,:]
        else:
            print('model is probably not a ViT')
            sys.exit(1)

def export_features(
        dataset:GeoDataset, 
        model:nn.Module,
        size:int,
        batch_size:int, 
        roi=None,
        cls_token=False,
        normalize=False,
        ):
    """
    Export the features outputed by a model to a numpy array.

    Args:
        dataset:GeoDataset
            The trochgeo dataset to infer on.
        model:nn.Module,
            The model to produce features
        size:int,
            Sampling size (in pixels). Also defines the stride for now.
        batch_size:int, 
        roi=None,
            Subregion to sample on the dataset
        cls_token=False,
            Whether or not to perform inference at cls_token level 
            (i.e. one output pixel per sample) or at patch_level (i.e. one output pixel per patch)
        normalize=False,
            Whether or not to remove the mean values of the tensors in the outputed array

    Returns:
        type and description of the returned object.macro_img, bboxes, patch_size
        macro_img: 
            Reconstructed output raster as a numpy array
        bboxes: 
            List of bounding boxes passed in the dataloader.
            Useful to get the top left corner afterward and reproject the array as a geotiff. 
        patch_size:
            size of the output pixels in native pixels. Also useful to reproject the array.
    """

    model.eval()
    model = model.cuda()

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

    # Initialize tensor used to store the projected feature space
    feat_img = None
    mean_tensor = None
    bboxes = []

    for i, batch in enumerate(dataloader):
        print(f"{i / len(dataloader):.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        images = images.type(torch.float)
        images = images.cuda()

        for sample in unbind_samples(batch):
            bboxes.append(sample['bbox'])

        with torch.no_grad():
            # feat = model(images)
            feat = get_features(model, images, cls_token)
            feat = feat.half()

            if i == 0 :
                if cls_token:
                    patch_size = images.shape[2]
                else:
                    n_patches = int(np.sqrt(feat.shape[1]))
                    patch_size = int(images.shape[2] / n_patches)

            if not cls_token:
                feat = feat.view(feat.shape[0],n_patches,n_patches,feat.shape[-1])

            if normalize:
                if mean_tensor is None:
                    mean_tensor = torch.sum(feat, dim=0)
                else:
                    mean_tensor = mean_tensor + torch.sum(feat, dim=0)

            feat = feat.detach().cpu().numpy()

            # Concatenate the transformed features to the existing tensor
            if feat_img is None:
                feat_img = feat
            else:
                feat_img = np.concatenate((feat_img, feat), axis=0)

    if normalize:
        mean_tensor = mean_tensor / len(bboxes)
        mean_tensor = mean_tensor.detach().cpu().numpy()

        feat_img = feat_img - mean_tensor

    del model

    Nx = len(bboxes)
    for i in range(1, len(bboxes)):
        if bboxes[i][0] < bboxes[i - 1][0]:
            Nx = i
            break
    Ny = int(len(bboxes) / Nx)
    print('\n')
    print(Nx, Ny)

    if cls_token:
        macro_img = reconstruct_img_cls(feat_img, Nx, Ny)
    else:
        macro_img = reconstruct_img_patch(feat_img, Nx, Ny)

    return macro_img, bboxes, patch_size



def reconstruct_img_patch(div_images, Nx, Ny):
    """
    recustruct numpy array of agregated patch features into a raster.

    Args:
        div_image:
            numpy array containing agregated features.
        Nx:
            Number of samples in a row in the GridGeoSampler
        Ny:
            Number of samples in a col in the GridGeoSampler

    Returns:
        aggregated_image:
            numpy array of shape (Ny*h,Nx*w,feature_dim) with h and w being the height and width of patches
    """
    image_shape = div_images.shape[1:]  # Shape of each tensor
    h, w, channels = image_shape
    
    reconstructed_height = h * Ny
    reconstructed_width = w * Nx
    
    if len(div_images.shape) == 2:
        return np.array([[div_images[Nx * (Ny - j - 1) + i] for i in range(Nx)] for j in range(Ny)])
    
    # Initialize the aggregated image
    aggregated_image = np.zeros((reconstructed_height, reconstructed_width, channels), dtype=np.float16)
    print(aggregated_image.shape)

    # Iterate over rows of the original grid
    for j in range(Ny):
        print(f"{j / Ny:.2%}", end="\r")
        # Iterate over columns of the original grid
        for i in range(Nx):
            idx = (Ny-1-j) * Nx + i
            if idx < len(div_images):
                x_start = i * w
                x_end = (i + 1) * w
                y_start = j * h
                y_end = (j + 1) * h
                aggregated_image[y_start:y_end, x_start:x_end, :] = div_images[idx]
    
    return aggregated_image


def reconstruct_img_cls(div_images, Nx, Ny):
    """
    recustruct numpy array of agregated cls_token features into a raster.

    Args:
        div_image:
            numpy array containing agregated features.
        Nx:
            Number of samples in a row in the GridGeoSampler
        Ny:
            Number of samples in a col in the GridGeoSampler

    Returns:
        aggregated_image:
            numpy array of shape (Ny,Nx,feature_dim).
    """
    channels = div_images.shape[1]  # Shape of each tensor
    
    # Initialize the aggregated image
    aggregated_image = np.zeros((Ny, Nx, channels), dtype=np.float16)

    # Iterate over rows of the original grid
    for j in range(Ny):
        print(f"{j / Ny:.2%}", end="\r")
        # Iterate over columns of the original grid
        for i in range(Nx):
            idx = (Ny-1-j) * Nx + i
            if idx < len(div_images):
                x_start = i 
                x_end = i + 1 
                y_start = j
                y_end = j + 1
                aggregated_image[y_start:y_end, x_start:x_end, :] = div_images[idx]
    
    return aggregated_image


def get_pixel_values_shp(raster_path, gdf):

    with rasterio.open(raster_path) as src:
        # Get raster metadata
        crs = src.crs
        raster_array = src.read()
        xmin, ymin, xmax, ymax = src.bounds
        ## TODO filter in xmin xmax etc...

        gdf = gdf.to_crs(crs)
        pixel_values = []
        for point in gdf.iterrows():
            index, data = point
            row, col = src.index(data.geometry.x, data.geometry.y)
            pixel_values.append(list(raster_array[:,row, col]))

        gdf['px_values'] = pixel_values

    return gdf


if __name__ == "__main__":

    
    ## Example workflow
    import os
    import tempfile
    from torchgeo.datasets import NAIP
    from torchgeo.datasets.utils import download_url
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    naip_root = os.path.join(tempfile.gettempdir(), "naip")
    proj_dir = os.path.join(tempfile.gettempdir(), "proj")
    proj_dir = os.path.expanduser(proj_dir)
    os.makedirs(proj_dir, exist_ok=True)

    naip_url = (
        "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
    )

    # tile may not download properly, leading to a `TIFFReadEncodedTile() failed` error
    # a simple wget with this url probably will solve the issue
    tile = "m_3807511_ne_18_060_20181104.tif"
    download_url(naip_url + tile, naip_root)

    dataset = NAIP(naip_root)

    MEANS = [122.39, 118.23, 98.1, 120.]
    SDS = [39.81, 37.33, 33.04, 30.]

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image"],
            )
    dataset.transforms = transform

    bb=dataset.bounds
    xlim = bb[0] + (bb[1]-bb[0])* 0.1
    xlim_max = bb[0] + (bb[1]-bb[0])* 0.5
    ylim = bb[2] + (bb[3]-bb[2])* 0.9
    ylim_max = bb[2] + (bb[3]-bb[2])* 0.1
    roi=BoundingBox(xlim, xlim_max, ylim_max, ylim, bb[4], bb[5])

    ## init via timm
    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True,in_chans=len(MEANS), num_classes=0)
    
    ## init via torch hub
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # model = vit_first_layer_with_nchan(model, in_chans=len(MEANS))

    size = 224

    ## export features to numpy
    feat_img, bboxes, patch_size= export_features(
            dataset, 
            model,
            size, 
            20, 
            roi=roi, 
            # cls_token=True,
            # normalize=False,
            )
    top_left = (bboxes[0].minx, bboxes[-1].maxy)
    array_to_geotiff(
            array=feat_img,
            output_file='./out/feat.tif',
            top_left_corner_coords=top_left,
            pixel_width=dataset.res * patch_size,
            pixel_height=dataset.res * patch_size,
            crs=dataset.crs,
            # dtype='int8', ## if clusters
            )

    gdf = gpd.read_file('/tmp/naip/template.shp')
    gdf = get_pixel_values_shp('./out/feat.tif', gdf)

    template_npy = np.asarray(list(gdf['px_values']))
    template = torch.from_numpy(template_npy)
    template = torch.mean(template, dim=0)

    # ## save resulting numpy if needed to avoid performing export twice
    # np.save('./feat_img.npy',feat_img)
    # feat_img = np.load('./feat_img.npy')
    feat_img = torch.from_numpy(feat_img)
    print(feat_img.shape)

    print('projections')

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    sim = cos(feat_img,template)
    print(sim.shape)
    sim = sim.unsqueeze(-1)
    sim = sim.numpy()
    array_to_geotiff(
            array=sim,
            output_file='./out/sim.tif',
            top_left_corner_coords=top_left,
            pixel_width=dataset.res * patch_size,
            pixel_height=dataset.res * patch_size,
            crs=dataset.crs,
            # dtype='int8', ## if clusters
            )










    ### e.g. PCA with 6 components on the features
    # pca = PCA(n_components=6)
    # tmp_img = pca.fit_transform(feat_img.reshape(-1, feat_img.shape[-1]))

    # ### UMAP after PCA
    # # umap = umap.UMAP(n_components=3)
    # # tmp_img = umap.fit_transform(tmp_img)

    # ### Kmeans after PCA
    # kmeans = KMeans(n_clusters=5)
    # tmp_img = kmeans.fit_transform(tmp_img) ## to kmeans transform
    # # tmp_img = kmeans.fit_predict(tmp_img) ## to kmeans labels
    # print(tmp_img.shape)

    # ## reshape to correct shape
    # feat_img = tmp_img.reshape((feat_img.shape[0], feat_img.shape[1],-1))

    # print('to geotiff')
    # top_left = (bboxes[0].minx, bboxes[-1].maxy)
    # array_to_geotiff(
    #         array=feat_img,
    #         output_file='./out/proj.tif',
    #         top_left_corner_coords=top_left,
    #         pixel_width=dataset.res * patch_size,
    #         pixel_height=dataset.res * patch_size,
    #         crs=dataset.crs,
    #         # dtype='int8', ## if clusters
    #         )

    # # go back to the original spatial resolution if needed
    # with rasterio.open(os.path.join(naip_root,tile)) as template_src:
    #     # Get the template resolution
    #     orig_resolution = template_src.transform[0]

    # change_tif_resolution(f'{proj_dir}/proj.tif',f'{proj_dir}/proj_rescale.tif', orig_resolution)