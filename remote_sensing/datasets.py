import re
import os
import sys
import glob
import functools
import rasterio
import torch
import numpy as np
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.io import DatasetReader
from torch import Tensor
from rasterio.windows import from_bounds
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast, Union
from torchgeo.datasets.utils import disambiguate_timestamp, BoundingBox
from torchgeo.datasets import RasterDataset, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple
from torch.utils.data import Dataset
import geopandas as gpd


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


class ROIDataset(RasterDataset):

    def __init__(
            self, 
            root: str,
            gdf: gpd.GeoDataFrame, 
            target_var: str, 
            target_indexes: Optional[List] = None,
            size: Union[Tuple[float, float], float] = 1,
            units: Units = Units.PIXELS,
            normalize_target_var=False,
            crs: Optional[CRS] = None,
            res: Optional[float] = None,
            bands: Optional[Sequence[str]] = None,
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            cache: bool = True,
            ):

        super().__init__(
                root,
                crs,
                res,
                bands,
                transforms,
                cache,
                )
        
        self.target_var = target_var
        self.size = _to_tuple(size)
        self.units = units
        # convert to meters
        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
        self.normalize = normalize_target_var
        print(self.size)
        self.target_indexes = target_indexes
        self.gdf = self._prepare_gdf(gdf)
        self.rois = self.gdf['bboxes']



    def _polygon_to_bbox(self, polygon):
        bounds = list(polygon.bounds)
        bounds[1], bounds[2] = bounds[2], bounds[1]
        # keeping temporal coordinates from raster dataset
        return BoundingBox(*bounds, self.index.bounds[4], self.index.bounds[5])
    
    def _get_intersected_bboxes(self, gdf):
        pathname = os.path.join(self.root, "**", self.filename_glob)
        file_list = []
        for filepath in glob.iglob(pathname, recursive=True):
            file_list.append(filepath)
        return gdf.loc[[intersects_with_img(gdf['bboxes'][i], file_list) for i in gdf.index]]


    def _prepare_gdf(self, gdf):
        # remove false geometries
        print("begin : ", len(gdf['geometry']))
        gdf = gdf.loc[gdf['geometry']!=None]
        print("geom : ", len(gdf['geometry']))
        gdf = gdf.to_crs(self.crs)
        gdf = gdf.drop_duplicates()
        print("drop_duplicates : ", len(gdf['geometry']))
        # remove nas in target variable
        gdf = gdf.dropna(subset=[self.target_var])
        print("dropna : ", len(gdf['geometry']))

        # if geodataframe has points, convert to square with buffer of self.size meters
        if gdf.geom_type.unique() == "Point":
            gdf.geometry = gdf.buffer(self.size[0], cap_style = 3)
        print("buffer : ", len(gdf['geometry']))

        # only conserves rois which intersect with the images from the dataset
        gdf['bboxes'] = [self._polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
        gdf = self._get_intersected_bboxes(gdf)
        print("intersect : ", len(gdf['geometry']))

        if self.normalize:
            gdf[self.target_var] = (gdf[self.target_var] - gdf[self.target_var].mean()) / gdf[self.target_var].std()

        gdf.index = [i for i in range(len(gdf))]
        if self.target_indexes is not None:
            gdf = gdf.iloc[self.target_indexes][self.target_var]
        print("Nb roi : ", len(gdf['bboxes']))

        return(gdf)

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        """Retrieve image/mask and metadata indexed by query.

        Args:
            index: Index of sample to fetch

        Returns:
            sample of image/mask, metadata and ground truth at that index

        Raises:
            IndexError: if query is not found in the index
        """
        query = self.gdf.iloc[idx]['bboxes']
        gt = self.gdf.iloc[idx][self.target_var]

        sample = super().__getitem__(query)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        sample["gt"] = gt

        return sample




class CustomLabelDataset(Dataset):
    def __init__(self, gdf, lab_column, transform=None):
        self.labels = gdf[lab_column]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        if self.transform:
            label = self.transform(label)
        return label
    
# class CustomDataset(RasterDataset):
#     filename_glob = "*.tif"
#     is_image = True


if __name__ == "__main__":
    

    gdf = gpd.read_file("~/congo/pointages/22_10_19_pointages.shp", driver='ESRI Shapefile')
    print(gdf)
    class CustomDataset(ROIDataset):
        filename_glob = "1959.tif"
    dataset = CustomDataset(
            root="/home/ptresson/congo/",
            gdf=gdf,
            target_var='C_ID',
            transforms=None,
            size=112,
            )
    sample = dataset.__getitem__(0)
    print(sample)
    print(dataset.__len__())
