from geopandas.io.sql import GeoDataFrame

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from numpy import int16
import numpy as np
import pandas as pd


if __name__ == "__main__":


    # ============================= SNR =============================
	# # Read in vector
    # pilote_org = gpd.read_file("~/sharaan/habitats_snr.gpkg", layer="pilot_site_habitat")
    # scale_up_org = gpd.read_file("~/sharaan/habitats_snr.gpkg", layer="scale_up_site_habitat")

    # pilote = GeoDataFrame()
    # scale_up = GeoDataFrame()

    # pilote['geometry'] = pilote_org['geometry']
    # pilote['name_habit'] = pilote_org['name_habit']
    # print(len(pilote['geometry']))
    # print(len(pilote['name_habit']))

    # scale_up['geometry'] = scale_up_org['geometry']
    # scale_up['name_habit'] = scale_up_org['hab_type']
    # print(len(scale_up['geometry'].dropna()))
    # print(len(scale_up['name_habit'].dropna()))

    # pilote = pilote.dropna()
    # scale_up = scale_up.dropna()
    # print(scale_up)

    # pilote = pilote.to_crs(scale_up.crs)

    # vector = gpd.GeoDataFrame(pd.concat([scale_up, pilote], ignore_index=True), crs=scale_up.crs)



    # ========================= hegra =====================
    vector = gpd.read_file("./hegra.gpkg", layer="habitat_hegra_paul")

    # Get list of geometries for all features in vector file
    geom = [shapes for shapes in vector.geometry]

	# Open example raster
    raster = rasterio.open(r"/home/ptresson/pteryx/pteryx/data/tifs/hegra.tif")

    classes = np.unique(vector['habitat_name'].dropna()) #drop na in classes
    nclasses = len(classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)} 

    vector['class_idx'] = [class_to_idx[cls_name] for cls_name in vector['habitat_name'].dropna()]

    vector = vector.to_crs(raster.crs)
    vector['class_idx'] = vector['class_idx'] + 1
	# create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value = ((geom,value) for geom, value in zip(vector.geometry, vector['class_idx']))
	# print(geom_value)
	# print(vector['class_idx'].unique())

	# Rasterize vector using the shape and transform of the raster
    rasterized = features.rasterize(geom_value,
                                    out_shape = raster.shape,
                                    transform = raster.transform,
                                    all_touched = True,
                                    fill = 0,   # background value
                                    merge_alg = MergeAlg.replace,
                                    dtype = int16)


    with rasterio.open(
            "out/hegra_mask.tif", "w",
            driver = "GTiff",
            crs = raster.crs,
            transform = raster.transform,
            dtype = rasterio.uint8,
            count = 1,
            width = raster.width,
            height = raster.height) as dst:
        dst.write(rasterized, indexes = 1)

