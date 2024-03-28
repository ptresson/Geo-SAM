# general
import os
import warnings
import joblib
from pathlib import Path

# pytorch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K

# torchgeo
from torchgeo.datasets import BoundingBox, RasterDataset, stack_samples, unbind_samples
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
import hdbscan
warnings.filterwarnings("ignore")

# custom modules
from utils import change_tif_resolution, load_pretrained_vit
from utils import vit_first_layer_with_nchan, reconstruct_img
from utils import numpy_to_geotif
from utils import export_on_map
from utils import get_mean_sd_by_band, get_crs
from utils import remove_black_tiles, remove_empty_tiles, remove_black_tiles2
from custom_datasets import *

from visualisation import fit_proj, inf_proj

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fit and infer a projection model on a geotiff dataset.")
    
    parser.add_argument("input_dir", type=str, help="Directory containing the original geotiff dataset.")
    parser.add_argument("output_geotiff", type=str, help="Path to save the output geotiff.")

    parser.add_argument("--input_glob", type=str, default='*.[Tt][Ii][Ff]', help="Glob to find the original geotiff dataset.")
    parser.add_argument("--model_weights", type=str, default=None, help="Path to the pretrained model weights.")
    
    # Arguments for fitting projection
    parser.add_argument("--fit_projection", action="store_true", default=True, help="Fit the projection model.")
    parser.add_argument("--projection_method", type=str, default="umap", help="Projection method (umap, pca, hdbscan, kmeans).")
    parser.add_argument("--projection_output", type=str, default="out/proj.pkl", help="Path to save the fitted projection model.")
    parser.add_argument("--cls_token", action="store_true", help="ViT specific. Is the projection fitted at image (cls_token) or patch (patch_token) level.")
    parser.add_argument("--feat_dim", type=int, default=1024, help="Dimension of the feature space output by the backbone.")
    parser.add_argument("--batch_size", type=int, default=10, help="How many samples per batch to load.")
    parser.add_argument("--exclude_value", type=float, default=None, help="Value to exclude tiles below the cumulative sum.")
    parser.add_argument("--nsamples", type=int, default=100, help="The number of samples to fit the projection or clustering algorithm.")
    parser.add_argument("--size", type=int, default=224, help="The size of the sampled images (in pixels).")
    parser.add_argument("--patch_h", type=int, default=16, help="The height of the patches used for analysis.")
    parser.add_argument("--patch_w", type=int, default=16, help="The width of the patches used for analysis.")
    parser.add_argument("--roi", nargs=4, type=float, help="The region of interest (ROI) for the analysis.")
    parser.add_argument("--n_components", type=int, default=3, help="The number of components for PCA or UMAP.")
    parser.add_argument("--n_neighbors", type=int, default=10, help="The number of neighbors for UMAP.")
    parser.add_argument("--min_samples", type=int, default=5, help="The minimum number of samples for HDBSCAN.")
    parser.add_argument("--min_cluster_size", type=int, default=100, help="The minimum cluster size for HDBSCAN.")
    parser.add_argument("--n_clusters", type=int, default=8, help="The number of clusters for KMeans.")

    # Arguments for inference
    parser.add_argument("--infer_projection", default=True, action="store_true", help="Infer the projection on the dataset.")
    parser.add_argument("--rescale_to_native", default=True, action="store_true", help="Have the output geotiff to the same spatial resolution as the input.")
    parser.add_argument("--npy_output_path", type=str, default=None, help="Path to save the output feature space.")
    
    return parser.parse_args()


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create RasterDataset
    class CustomRaster(RasterDataset):
        filename_glob=args.input_glob

    dataset = CustomRaster(args.input_dir)
    tif_files = [f for f in os.listdir(args.input_dir) if f.endswith(args.input_glob)]
    tif_files = [file for file in Path(args.input_dir).glob(args.input_glob) if not file.stem.endswith("mask")]
    means, sds = get_mean_sd_by_band(tif_files[0])

    # take pretrained model and adapt number of input bands if needed
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if args.model_weights:
        model = load_pretrained_vit(
                model, 
                checkpoint_path=args.model_weights, 
                nchannels=len(means)
                )
    model = vit_first_layer_with_nchan(model, in_chans=len(means))

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(means,sds), # normalize occurs only on raster, not mask
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image"],
            )
    dataset.transforms = transform

    model.to(device)

    if args.fit_projection:
        print("Fit projection\n")
        fit_proj(
                dataset,
                model,
                method=args.projection_method,
                cls_token=args.cls_token,
                feat_dim=args.feat_dim,
                batch_size=args.batch_size,
                exclude_value=args.exclude_value,
                nsamples=args.nsamples,
                size=args.size,
                patch_w=args.patch_w, 
                patch_h=args.patch_h,
                roi=args.roi,
                n_components=args.n_components,
                n_neighbors=args.n_neighbors,
                min_samples=args.min_samples,
                min_cluster_size=args.min_cluster_size,
                n_clusters=args.n_clusters,
                out_path_proj_model=args.projection_output
                )

    if args.infer_projection:
        print("Projection inference\n")
        macro_img, bboxes = inf_proj(
                dataset,
                model,
                feat_dim=args.feat_dim,
                cls_token=args.cls_token,
                out_dim=args.n_components,
                batch_size=args.batch_size,
                size=args.size,
                patch_w=args.patch_w, 
                patch_h=args.patch_h,
                roi=args.roi,
                path_proj_model=args.projection_output,
                path_out=args.output_geotiff,
                )

        if args.npy_output_path:
            np.save(args.npy_output_path, macro_img)

        # Save the output geotiff
        if args.cls_token:
            numpy_to_geotif(
                original_image=tif_files[0],
                numpy_image=macro_img,
                dtype='float32',
                pixel_scale=args.size,
                out_path=args.output_geotiff
            )
        else:
            numpy_to_geotif(
                original_image=tif_files[0],
                numpy_image=macro_img,
                dtype='float32',
                pixel_scale=int(args.size/args.patch_w),
                out_path=args.output_geotiff
            )

    if args.rescale_to_native:
        with rasterio.open(tif_files[0]) as src:
            # Get the template resolution
            orig_resolution = src.transform[0]

        # overwrites the output_file
        change_tif_resolution(args.output_geotiff,args.output_geotiff, orig_resolution)

if __name__ == "__main__":
    args = parse_args()
    main(args)

