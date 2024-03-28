import joblib
import sys
import warnings

# pytorch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import timm

# torchgeo
from torchgeo.datasets import BoundingBox, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd
import rasterio

# data
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import KMeans
# import hdbscan
warnings.filterwarnings("ignore")

# custom modules
from remote_sensing.utils import change_tif_resolution
from remote_sensing.utils import vit_first_layer_with_nchan
from remote_sensing.utils import array_to_geotiff
from remote_sensing.utils import export_on_map
from remote_sensing.utils import get_mean_sd_by_band, get_crs
from remote_sensing.utils import remove_black_tiles, remove_empty_tiles, remove_black_tiles2
from custom_datasets import *


def fit_proj(
        dataset,
        model,
        cls_token=False,
        feat_dim=1024,
        batch_size=10,
        exclude_value=-77.04,
        nsamples=100,
        size=224,
        patch_h=100,
        patch_w=100,
        method='pca',
        out_path_proj_model="out/proj.pkl",
        roi=None,
        n_components=3,
        n_neighbors=10,
        min_samples=5,
        min_cluster_size=100,
        n_clusters=8,
        ):
    """
    Fit a projection or clustering algorithm in the feature space of a deep learning model.

    Args:
        dataset (RasterDataset): Torchgeo raster dataset containing the data.
        model: PyTorch model to use for projection into the feature space.
        cls_token (bool, optional): ViT specific. Is the projection fitted at image (cls_token) or patch (patch_token) level. `cls_token=False` will default to use at the patch token level.
            (default=False).
        feat_dim (int, optional): Dimension of the feature space output by the backbone. This value depends on the chosen architecture.
            (default=1024).
        batch_size (int, optional): how many samples per batch to load
            (default: 10).
        exclude_value (float, optional): If a value is provided, tiles whose cumulated sum is below that value will be removed.
            (default: -77.04)
        nsamples (int, optional): The number of samples to fit the projection or clustering algorithm.
            (default: 100)
        size (int, optional): The size of the sampled images (in pixels).
            (default: 224)
        patch_h (int, optional): The height of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        patch_w (int, optional): The width of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        method (str, optional): The method used for fitting the projection or clustering. Choose between 'pca', 'umap', 'hdbscan', and 'kmeans'.
            (default: 'pca')
        out_path_proj_model (str, optional): The path to save the fitted projection or clustering model to a .pkl file.
            (default: "out/proj.pkl")
        roi (tuple, optional): The region of interest (ROI) for the analysis. By default, it takes the entire dataset bounds.
            (default: None)
        n_components (int, optional): The number of components for PCA or UMAP.
            (default: 3)
        n_neighbors (int, optional): The number of neighbors for UMAP.
            (default: 10)
        min_samples (int, optional): The minimum number of samples for HDBSCAN.
            (default: 5)
        min_cluster_size (int, optional): The minimum cluster size for HDBSCAN.
            (default: 100)
        n_clusters (int, optional): The number of clusters for KMeans.
            (default: 8)
    """

    # If ROI is smaller than the full dataset
    if roi:
        sampler = RandomGeoSampler(dataset, size=size, length=nsamples, roi=roi)
    else:
        sampler = RandomGeoSampler(dataset, size=size, length=nsamples)

    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        collate_fn=stack_samples, 
        shuffle=False, 
        batch_size=batch_size
    )

    feat_img = None  # Initialize the feature tensor

    i = 0
    for batch in dataloader:
        print(f"{i / len(dataloader):.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        bboxes = batch['bbox']
        images = images.type(torch.float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)

        do_forward = True

        # if the tensor contains the exclue value, it is not processed.
        # the purpose is to handle black or patially empty tiles
        if exclude_value:
            images, _, is_non_empty = remove_black_tiles2(images, bboxes, exclude_value)
            if not is_non_empty:
                do_forward = False

        if do_forward:
            with torch.no_grad():
                if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
                    features_dict = model.forward_features(images)

                    if cls_token:
                        try:
                            feat = features_dict['x_norm_clstoken']
                        except (KeyError, IndexError):
                            feat = features_dict
                        feat = feat.detach().cpu().numpy()
                        if feat_img is None:
                            feat_img = feat
                        feat_img = np.concatenate((feat_img, feat), axis=0)
                    else:

                        try:
                            feat = features_dict['x_norm_patchtokens']
                        except KeyError:
                            raise NotImplementedError("model architecture does not support patch_tokens inference")

                        feat = feat.detach().cpu().numpy()
                        feat = feat.reshape(images.shape[0] * patch_h * patch_w, feat_dim)
                        if feat_img is None:
                            feat_img = feat
                        feat_img = np.concatenate((feat_img, feat), axis=0)
                else:
                    feat = model(images)
                    if feat_img is None:
                        feat_img = feat
                    feat_img = np.concatenate((feat_img, feat), axis=0)

        i += 1

    # Using different methods to fit projection/clustering on the feature space.
    # Each method comes with its own hyperparameters.
    if method == 'pca':
        embedder = PCA(n_components=n_components)
    elif method == 'umap':
        embedder = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    elif method == 'hdbscan':
        embedder = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            prediction_data=True
        )
    elif method == 'kmeans':
        embedder = KMeans(n_clusters=n_clusters)

    embedder = embedder.fit(feat_img)

    # Saving to a .pkl file
    joblib.dump(embedder, out_path_proj_model)


def inf_proj(
        dataset,
        model,
        cls_token=False,
        feat_dim=1024,
        out_dim=3,
        batch_size=10,
        size=224,
        patch_h=100,
        patch_w=100,
        path_proj_model="out/proj.pkl",
        roi=None,
        path_out=None,
        ):
    """
    Perform inference using a pre-fitted projection model on the feature space of a deep learning model.

    Args:
        dataset (RasterDataset): Torchgeo raster dataset containing the data.
        model: PyTorch model to use for feature extraction and inference.
        cls_token (bool, optional): ViT specific. If True, works at cls_token (image) scale; otherwise, works at patch scale.
            (default=False).
        out_dim (int, optional): Dimension of the output feature space after projection.
            (default: 3)
        feat_dim (int, optional): Dimension of the feature space output by the backbone. This value depends on the chosen architecture.
            (default=1024).
        batch_size (int, optional): How many samples per batch to load.
            (default: 10).
        size (int, optional): The size of the tiles fed to the model (in pixels).
            (default: 224)
        patch_h (int, optional): The height of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        patch_w (int, optional): The width of the patches used for analysis. It depends on the desired spatial resolution.
            (default: 100)
        path_proj_model (str, optional): The path to the pre-fitted projection model stored in a .pkl file.
            (default: "out/proj.pkl")
        roi (tuple, optional): The region of interest (ROI) for the analysis. By default, it takes the entire dataset bounds.
            (default: None)
        path_out (str, optional): The path to save the output feature space. If None, the output is not saved.
            (default: None)

    Returns:
        tuple: A tuple containing the macro image of the feature space and the bounding boxes of the processed samples.
    """

    # Load pre-fitted projection or clustering model
    embedder = joblib.load(path_proj_model)

    # Set up data loader based on the specified region of interest (ROI)
    if roi:
        sampler = GridGeoSampler(dataset, size=size, stride=size, roi=roi)
    else:
        sampler = GridGeoSampler(dataset, size=size, stride=size)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=stack_samples,
        shuffle=False,
        batch_size=batch_size
    )

    N = len(dataloader)
    bboxes = []  # List to store bounding boxes of processed samples

    # Initialize tensor used to store the projected feature space
    feat_img = None

    i = 0
    for batch in dataloader:
        print(f"{i / N:.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        images = images.type(torch.float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)

        # Extract bounding boxes for each sample
        for sample in unbind_samples(batch):
            bboxes.append(sample['bbox'])

        with torch.no_grad():
            features_dict = model.forward_features(images)

            if cls_token:
                feat = features_dict['x_norm_clstoken']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(images.shape[0], feat_dim)
            else:
                feat = features_dict['x_norm_patchtokens']
                feat = feat.detach().cpu().numpy()
                feat = feat.reshape(images.shape[0] * patch_h * patch_w, feat_dim)

            # Transform features using the pre-fitted embedder model
            red_features = embedder.transform(feat)

            # Reshape the transformed features to match the desired output shape
            if cls_token:
                red_features = red_features.reshape(images.shape[0], out_dim)
            else:
                red_features = red_features.reshape(images.shape[0], patch_h, patch_w, out_dim)

            # Concatenate the transformed features to the existing tensor
            if feat_img is None:
                feat_img = red_features
            else:
                feat_img = np.concatenate((feat_img, red_features), axis=0)

        i += 1

    # Determine the grid dimensions based on bounding boxes
    Nx = len(bboxes)
    for i in range(1, len(bboxes)):
        if bboxes[i][0] < bboxes[i - 1][0]:
            Nx = i
            break
    Ny = int(len(bboxes) / Nx)

    # Generate a macro image from the projected feature space
    macro_img = reconstruct_img(feat_img, Nx, Ny)

    # Save the macro image if the output path is specified
    if path_out:
        np.save(path_out, macro_img)

    return macro_img, bboxes




def inf_cluster(
        dataset,
        model,
        batch_size = 10, 
        exclude_value=None,
        size=224, # size of sampled images
        path_proj_model= "out/proj.pkl",
        roi=None, # defaults takes all dataset bounds
        ):
    """
    This function may be depreciated in the sense that performing a clustering on a reprojected geotiff
    will be more reliable and simple.
    Moreover, it only works at cls_token level. First doing a projection and then cluster allows to works
    at patch_token level.
    """

    clusterer = joblib.load(path_proj_model)

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
    labels = []

    i=0
    for batch in dataloader:
        print(f"{i/N:.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4: # transforms can create a false dim ?
            images = images.squeeze(1)
        batch_bboxes = batch['bbox']
        images = images.type(torch.float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)

        do_forward = True
        if exclude_value:
            images, batch_bboxes, is_non_empty = remove_black_tiles2(images, batch_bboxes, exclude_value)
            if is_non_empty == False:
                do_forward = False
        
        if do_forward:

            bboxes.extend(batch_bboxes)

            with torch.no_grad():

                features_dict = model.forward_features(images)
                try:
                    feat = features_dict['x_norm_clstoken']
                except:
                    feat = features_dict
                feat = feat.detach().cpu().numpy()

                try :
                    label = clusterer.predict(feat)
                # hdbscan prediction do not work the same way as classic kmeans
                except:
                    label, strengths = hdbscan.prediction.approximate_predict(
                                clusterer,
                                feat
                                )
                    label = list(label)

                label = label.astype('int8')
                labels.extend(label)

                del feat
                del features_dict
                del batch_bboxes
        i+=1

    return labels, bboxes


def remove_zero_row_cols(arr):
    # Find rows and columns with non-zero elements
    non_zero_rows = np.any(arr != 0, axis=(1, 2))
    non_zero_cols = np.any(arr != 0, axis=(0, 2))

    # Extract non-zero region
    result = arr[non_zero_rows][:, non_zero_cols]
    return result


# def get_features(model, images, cls_token):

#     if cls_token:
#         if hasattr(model, 'forward_head') and callable(getattr(model, 'forward_head')):
#             return model.forward_head(images)
#         if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
#             features = model.forward_features(images)
#             return features['x_norm_clstoken']
#         else:
#             return model(images)
#     else:
#         if hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features')):
#             features = model.forward_features(images)
#             if isinstance(features, dict):
#                 return features['x_norm_patchtokens']
#             else:
#                 return features
#         else:
#             print('model is probably not a ViT')
#             sys.exit(1)


def get_features(model, images, cls_token):

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
        dataset, 
        model,
        size, 
        batch_size, 
        roi=None,
        cls_token=False,
        normalize=True,
        ):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to("cpu")
    #model = model.cpu() #surement modifier avec un if(available) pour faire tourner en local

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)

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

    #Means = get_sd_means...


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
    # ylim = bb[2] + (bb[3]-bb[2])* 0.2
    ylim_max = bb[2] + (bb[3]-bb[2])* 0.1
    roi=BoundingBox(xlim, xlim_max, ylim_max, ylim, bb[4], bb[5])

    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True,in_chans=len(MEANS), num_classes=0)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    size = 224
    macro_img, bboxes, patch_size= export_features(
            dataset, 
            model,
            size, 
            20, 
            roi=roi, 
            # cls_token=True,
            # normalize=False
            )
    # macro_img = remove_zero_row_cols(macro_img)
    print(macro_img.shape)

    print('PCA')
    pca = PCA(6) # take 3 principal components.
    pca_img = pca.fit_transform(macro_img.reshape(-1, macro_img.shape[-1]))
    # umap = umap.UMAP(n_components=3)
    # pca_img = umap.fit_transform(pca_img)
    kmeans = KMeans(n_clusters=5)
    pca_img = kmeans.fit_transform(pca_img)
    # pca_img = kmeans.fit_predict(pca_img)
    print(pca_img.shape)

    macro_img = pca_img.reshape((macro_img.shape[0], macro_img.shape[1],-1))

    print('to geotiff')
    top_left = (bboxes[0].minx, bboxes[-1].maxy)
    array_to_geotiff(
            array=macro_img,
            output_file='./featup.tif',
            top_left_corner_coords=top_left,
            pixel_width=dataset.res * patch_size,
            pixel_height=dataset.res * patch_size,
            # pixel_width=dataset.res,
            # pixel_height=dataset.res,
            crs=dataset.crs,
            # dtype='int8',
            )



    # numpy_to_geotif(
    #         original_image=os.path.join(naip_root,tile),
    #         numpy_image=macro_img,
    #         dtype='float32',
    #         pixel_scale=1,
    #         out_path=f'./featup.tif',
    #         n_channels=macro_img.shape[-1],
    #         )
    sys.exit(1)

    """
    example, fit a UMAP projection of this dataset through the model.
    The 768 dimension of the models feature space are reduced to 3 dimensions.
    5 neighbors are used for computation meaning a more local approximation of the manifold
    but less computation time that the default 15 (values should typically be between 2 and 100)
    Only 100 random samples are used to fit the projection rather than the entire dataset
    depending on available RAM and CPU, UMAP computation may take a while

    """

    print("Fit projection\n")
    fit_proj(
            dataset,
            model,
            size=size,
            nsamples=100,
            feat_dim=768,
            exclude_value=None,
            patch_w=16, 
            patch_h=16,
            method='umap',
            n_neighbors=20,
            cls_token=False,
            roi=None,
            batch_size=50,
            out_path_proj_model=f'{proj_dir}/proj.pkl'
            )


    """
    Inference of the fitted projection on the datatset following a grid.
    This results to a numpy array with pixels that are encompassing 
    several pixels in the original image: 
        - 16x16 pixels when working at patch_token level for a ViT with 16x16 patches
        - 224x224 when working at cls_token level
        
    """

    # Here an example performing inference only 
    # on the top left corner of the original image
    print(dataset.bounds)
    bb=dataset.bounds
    xlim = bb[0] + (bb[1]-bb[0])*0.25
    ylim = bb[2] + (bb[3]-bb[2])*0.25
    roi=BoundingBox(bb[0], xlim, bb[2], ylim, bb[4], bb[5])

    print("Projection inference\n")
    macro_img, bboxes = inf_proj(
            dataset,
            model,
            roi=roi,
            size=size,
            patch_w=16, 
            patch_h=16,
            cls_token=False,
            path_proj_model=f'{proj_dir}/proj.pkl',
            batch_size=500,
            )

    ## can be usefull to save progress
    # np.save(f'{proj_dir}/proj.npy', macro_img)

    numpy_to_geotif(
            original_image=os.path.join(naip_root,tile),
            numpy_image=macro_img,
            dtype='float32',
            pixel_scale=16,
            out_path=f'{proj_dir}/proj.tif'
            )

    # go back to the original spatial resolution if needed
    with rasterio.open(os.path.join(naip_root,tile)) as template_src:
        # Get the template resolution
        orig_resolution = template_src.transform[0]

    change_tif_resolution(f'{proj_dir}/proj.tif',f'{proj_dir}/proj_rescale.tif', orig_resolution)

