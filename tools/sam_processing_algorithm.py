import os
import time
import tifffile
from sklearn.decomposition import PCA
from torchgeo.datasets import BoundingBox, stack_samples, unbind_samples
#from remote_sensing.utils import array_to_geotiff
import subprocess
#from remote_sensing.visualisation import reconstruct_img_patch
from typing import Dict, Any, List
from pathlib import Path
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.gui import QgsDockWidget, QgsFileWidget
from qgis.utils import iface
from qgis.core import (QgsProcessing, Qgis,
                       QgsGeometry,
                       QgsRectangle,
                       QgsCoordinateReferenceSystem,
                       QgsUnitTypes,
                       QgsRasterBandStats,
                       QgsCoordinateTransform,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingFeedback,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterString,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterScale,
                       QgsProcessingParameterExpression,
                       QgsProcessingParameterRange,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterFeatureSink)
from qgis import processing
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
import timm
import torch
import sys
import os
import subprocess
from .torchgeo_sam import SamTestGridGeoSampler, SamTestRasterDataset
from torchgeo.samplers import Units
from torchgeo.datasets import BoundingBox, stack_samples
from torch.utils.data import DataLoader
import rasterio
import numpy as np
import pandas as pd
from torch import Tensor
import hashlib
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from ..ui.icons import QIcon_EncoderTool
from ..docs import encoder_help

# 0 for meters, 6 for degrees, 9 for unknown
UNIT_METERS = 0
UNIT_DEGREES = 6


list_features = []
#conda_environment = "ojala"
#activate_command = f"conda activate {conda_environment}"
#subprocess.run(activate_command, shell = True)

#eval "$(conda shell.bash hook)"
#conda activate ojala

#Coucou


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


def reconstruct_img_feat(div_images, Nx, Ny):
    image_shape = div_images.shape[2:]  # Shape of each tensor
    div_image_red = np.squeeze(div_images, axis = 1)
    
    channels, h, w = image_shape
    
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
                #aggregated_image[y_start:y_end, x_start:x_end, :] = div_images[idx].transpose(1, 2, 0)
                aggregated_image[y_start:y_end, x_start:x_end, :] = div_image_red[idx].transpose(1, 2, 0)
                #aggregated_image[y_start:y_end, x_start:x_end, :] = div_images[idx]
    
    return aggregated_image

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

def sam_first_layer_with_nchan(model, in_chans=1):

    kernel_size = model.image_encoder.patch_embed.proj.kernel_size
    stride = model.image_encoder.patch_embed.proj.stride
    embed_dim = model.image_encoder.patch_embed.proj.out_channels # corresponds to embed_dim
    # copy the original patch_embed.proj config 
    # except the number of input channels
    new_conv = torch.nn.Conv2d(
            in_chans, 
            out_channels=embed_dim,
            kernel_size=kernel_size, 
            stride=stride
            )
    # copy weigths and biases
    weight = model.image_encoder.patch_embed.proj.weight.clone()
    bias = model.image_encoder.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.image_encoder.patch_embed.proj = new_conv

    return model

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

class SamProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    # INPUT = 'INPUT'
    # CKPT = 'CKPT'
    # MODEL_TYPE = 'MODEL_TYPE'
    # BANDS = 'BANDS'
    # STRIDE = 'STRIDE'
    # EXTENT = 'EXTENT'
    # LOAD = 'LOAD'
    # OUTPUT = 'OUTPUT'
    # RANGE = 'RANGE'
    # RESOLUTION = 'RESOLUTION'
    # CRS = 'CRS'
    # CUDA = 'CUDA'
    # BATCH_SIZE = 'BATCH_SIZE'
    # CUDA_ID = 'CUDA_ID'

    FEAT_OPTION= 'FEAT_OPTION'
    INPUT = 'INPUT'
    CKPT = 'CKPT'
    MODEL_TYPE = 'MODEL_TYPE'
    BANDS = 'BANDS'
    STRIDE = 'STRIDE'
    EXTENT = 'EXTENT'
    LOAD = 'LOAD'
    OUTPUT = 'OUTPUT'
    RANGE = 'RANGE'
    RESOLUTION = 'RESOLUTION'
    CRS = 'CRS'
    CUDA = 'CUDA'
    BATCH_SIZE = 'BATCH_SIZE'
    CUDA_ID = 'CUDA_ID'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        cwd = Path(__file__).parent.parent.absolute()

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr(
                    'Input raster layer or image file path'),
            defaultValue=os.path.join(cwd,'rasters','test_multi.tif'),
            ),
        )



        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr(
                    'Select no more than 3 bands (preferably in RGB order, default to first 3 available bands)'),
                defaultValue=[1, 2, 3, 4, 5, 6, 7 ,8 , 9, 10, 11],
                parentLayerParameterName=self.INPUT,
                optional=True,
                allowMultiple=True,
            )
        )

        crs_param = QgsProcessingParameterCrs(
            name=self.CRS,
            description=self.tr('Target CRS (default to original CRS)'),
            optional=True,
        )

        res_param = QgsProcessingParameterNumber(
            name=self.RESOLUTION,
            description=self.tr(
                'Target resolution in meters (default to native resolution)'),
            type=QgsProcessingParameterNumber.Double,
            optional=True,
            minValue=0,
            maxValue=100000
        )

        # expression for scaling the raster values to [0,255]
        range_param = QgsProcessingParameterRange(
            name=self.RANGE,
            description=self.tr(
                'Data value range to be rescaled to [0, 255] (default to [min, max] of the values)'),  # inside processing extent
            type=QgsProcessingParameterNumber.Double,
            defaultValue=None,
            optional=True
        )

        cuda_id_param = QgsProcessingParameterNumber(
            name=self.CUDA_ID,
            # large images will be sampled into patches in a grid-like fashion
            description=self.tr(
                'CUDA Device ID (choose which GPU to use, default to device 0)'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=9
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr(
                    'Processing extent (default to the entire image)'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    'Stride (large image will be sampled into overlapped patches)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
                minValue=1,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr(
                    'SAM checkpoint path (download in advance)'),
                defaultValue = os.path.join(cwd,'checkpoint','sam_vit_l_0b3195.pth'),
                extension='pth',
            )
        ) 

        self.model_type_options = ['vit_h', 'vit_l', 'vit_b']
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.MODEL_TYPE,
                description=self.tr(
                    'SAM model type (b for base, l for large, h for huge)'),
                options=self.model_type_options,
                defaultValue=1,  # 'vit_l'
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"),
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA,
                self.tr("Use GPU if CUDA is available."),
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FEAT_OPTION,
                self.tr("Display features map"),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    'Batch size (take effect if choose to use GPU and CUDA is available)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD,
                self.tr("Load output features in Geo-SAM tool after processing"),
                defaultValue=True
            )
        )

        for param in (crs_param, res_param, range_param, cuda_id_param):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)

        # self.addOutput()

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        self.iPatch = 0
        
        self.feature_dir = ""
        
        self.FEAT_OPTION = self.parameterAsBoolean(
            parameters, self.FEAT_OPTION, context)

        feedback.pushInfo(
                f'PARAMETERS :\n{parameters}')
        
        feedback.pushInfo(
                f'CONTEXT :\n{context}')
        
        feedback.pushInfo(
                f'FEEDBACK :\n{feedback}')

        rlayer = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))

        self.selected_bands = self.parameterAsInts(
            parameters, self.BANDS, context)

        if len(self.selected_bands) == 0:
            max_band = min(3, rlayer.bandCount())
            self.selected_bands = list(range(1, max_band+1))

        # if len(self.selected_bands) > 3:
        #     raise QgsProcessingException(
        #         self.tr("Please choose no more than three bands!")
        #     )
        if max(self.selected_bands) > rlayer.bandCount():
            raise QgsProcessingException(
                self.tr("The chosen bands exceed the largest band number!")
            )

        ckpt_path = self.parameterAsFile(
            parameters, self.CKPT, context)
        model_type_idx = self.parameterAsEnum(
            parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(
            parameters, self.STRIDE, context)
        res = self.parameterAsDouble(
            parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(
            parameters, self.CRS, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context)
        self.load_feature = self.parameterAsBoolean(
            parameters, self.LOAD, context)
        self.use_gpu = self.parameterAsBoolean(
            parameters, self.CUDA, context)
        batch_size = self.parameterAsInt(
            parameters, self.BATCH_SIZE, context)
        range_value = self.parameterAsRange(
            parameters, self.RANGE, context)
        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)
        self.cuda_id = self.parameterAsInt(
            parameters, self.CUDA_ID, context)

        rlayer_data_provider = rlayer.dataProvider()

        # handle crs
        if crs is None or not crs.isValid():
            crs = rlayer.crs()
            # feedback.pushInfo(
            #     f'Layer CRS unit is {crs.mapUnits()}')  # 0 for meters, 6 for degrees, 9 for unknown
            # feedback.pushInfo(
            #     f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
            # if crs.mapUnits() == Qgis.DistanceUnit.Degrees:
            #     crs = self.estimate_utm_crs(rlayer.extent())

        # target crs should use meters as units
        # if crs.mapUnits() != Qgis.DistanceUnit.Meters:
        #     feedback.pushInfo(
        #         f'Layer CRS unit is {crs.mapUnits()}')
        #     feedback.pushInfo(
        #         f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
        #     raise QgsProcessingException(
        #         self.tr("Only support CRS with the units as meters")
        #     )

        if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
            layer_units = 'degrees'
        else:
            layer_units = 'meters'
        # if res is not provided, get res info from rlayer
        if np.isnan(res) or res == 0:
            res = rlayer.rasterUnitsPerPixelX()  # rasterUnitsPerPixelY() is negative
            target_units = layer_units
        else:
            # when given res in meters by users, convert crs to utm if the original crs unit is degree
            if crs.mapUnits() != UNIT_METERS: # Qgis.DistanceUnit.Meters:
                if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
                    # estimate utm crs based on layer extent
                    crs = self.estimate_utm_crs(rlayer.extent())
                else:
                    raise QgsProcessingException(
                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported.")
            target_units = 'meters'
            # else:
            #     res = (rlayer_extent.xMaximum() -
            #            rlayer_extent.xMinimum()) / rlayer.width()
        self.res = res

        # handle extent
        if extent.isNull():
            extent = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)
            extent_crs = rlayer.crs()
        else:
            if extent.isEmpty():
                raise QgsProcessingException(
                    self.tr("The extent for processing can not be empty!"))
            extent_crs = self.parameterAsExtentCrs(
                parameters, self.EXTENT, context)
        # if extent crs != target crs, convert it to target crs
        if extent_crs != crs:
            transform = QgsCoordinateTransform(
                extent_crs, crs, context.transformContext())
            # extent = transform.transformBoundingBox(extent)
            # to ensure coverage of the transformed extent
            # convert extent to polygon, transform polygon, then get boundingBox of the new polygon
            extent_polygon = QgsGeometry.fromRect(extent)
            extent_polygon.transform(transform)
            extent = extent_polygon.boundingBox()
            extent_crs = crs

        # check intersects between extent and rlayer_extent
        if rlayer.crs() != crs:
            transform = QgsCoordinateTransform(
                rlayer.crs(), crs, context.transformContext())
            rlayer_extent = transform.transformBoundingBox(
                rlayer.extent())
        else:
            rlayer_extent = rlayer.extent()
        if not rlayer_extent.intersects(extent):
            raise QgsProcessingException(
                self.tr("The extent for processing is not intersected with the input image!"))

        model_type = self.model_type_options[model_type_idx]
        if model_type not in os.path.basename(ckpt_path):
            raise QgsProcessingException(
                self.tr("Model type does not match the checkpoint"))

        img_width_in_extent = round(
            (extent.xMaximum() - extent.xMinimum())/self.res)
        img_height_in_extent = round(
            (extent.yMaximum() - extent.yMinimum())/self.res)
        # handle value range
        if (not np.isnan(range_value[0])) and (not np.isnan(range_value[1])):
            feedback.pushInfo(
                f'Input data value range to be rescaled: {range_value} (set by user)')
        else:
            if extent_crs == rlayer.crs():
                stat_extent = extent
            else:
                transform = QgsCoordinateTransform(
                    extent_crs, rlayer.crs(), context.transformContext())
                stat_extent = transform.transformBoundingBox(
                    extent)
            start_time = time.time()
            # set sample size to limit statistic time
            sample_size = min(1e8, img_height_in_extent*img_width_in_extent)
            min_values = []
            max_values = []
            for band in self.selected_bands:
                band_stats = rlayer_data_provider.bandStatistics(
                    bandNo=band, stats=QgsRasterBandStats.All, extent=stat_extent, sampleSize=sample_size)
                min_values.append(band_stats.minimumValue)
                max_values.append(band_stats.maximumValue)
            range_value[0] = min(min_values)
            range_value[1] = max(max_values)
            end_time = time.time()
            elapsed_time = (end_time - start_time)
            feedback.pushInfo(
                f'Input data value range to be rescaled: {range_value} (automatically set based on min-max value of input image inside the processing extent.)')
            feedback.pushInfo(
                f'Band statistics took time {elapsed_time:.3f}s'
            )

        if range_value[0] >= range_value[1]:
            raise QgsProcessingException(
                self.tr("Data value range is wrongly set or the image is with constant values."))

        # Send some information to the user
        feedback.pushInfo(
            f'Layer path: {rlayer_data_provider.dataSourceUri()}')
        # feedback.pushInfo(
        #     f'Layer band scale: {rlayer_data_provider.bandScale(self.selected_bands[0])}')
        feedback.pushInfo(f'Layer name: {rlayer.name()}')
        if rlayer.crs().authid():
            feedback.pushInfo(f'Layer CRS: {rlayer.crs().authid()}')
        else:
            feedback.pushInfo(
                f'Layer CRS in WKT format: {rlayer.crs().toWkt()}')
        feedback.pushInfo(
            f'Layer pixel size: {rlayer.rasterUnitsPerPixelX()}, {rlayer.rasterUnitsPerPixelY()} {layer_units}')

        feedback.pushInfo(f'Bands selected: {self.selected_bands}')

        if crs.authid():
            feedback.pushInfo(f'Target CRS: {crs.authid()}')
        else:
            feedback.pushInfo(f'Target CRS in WKT format: {crs.toWkt()}')
        # feedback.pushInfo('Band number is {}'.format(rlayer.bandCount()))
        # feedback.pushInfo('Band name is {}'.format(rlayer.bandName(1)))
        feedback.pushInfo(f'Target resolution: {self.res} {target_units}')
        # feedback.pushInfo('Layer display band name is {}'.format(
        #     rlayer.dataProvider().displayBandName(1)))
        feedback.pushInfo(
            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))
        feedback.pushInfo(
            (f'Processing image size: (width {img_width_in_extent}, '
             f'height {img_height_in_extent})'))

        # feedback.pushInfo(
        #     f'SAM Image Size: {self.sam_model.image_encoder.img_size}')

        rlayer_path = rlayer.dataProvider().dataSourceUri()
        rlayer_dir = os.path.dirname(rlayer_path)
        rlayer_name = os.path.basename(rlayer_path)

        # get mean and sd of dataset from raster metadata
        MEANS, SDS = get_mean_sd_by_band(rlayer_path)
        # subset with selected_bands
        MEANS = [MEANS[i-1] for i in self.selected_bands]
        SDS = [SDS[i-1] for i in self.selected_bands]

        SamTestRasterDataset.filename_glob = rlayer_name
        SamTestRasterDataset.all_bands = [
            rlayer.bandName(i_band) for i_band in range(1, rlayer.bandCount()+1)
        ]
        # currently only support rgb bands
        input_bands = [rlayer.bandName(i_band)
                       for i_band in self.selected_bands]
        # # ensure only three bands are used, less than three bands will be broadcasted to three bands
        # input_bands = (input_bands * 3)[0:3]

        if crs == rlayer.crs():
            rlayer_ds = SamTestRasterDataset(
                root=rlayer_dir, crs=None, res=self.res, bands=input_bands, cache=False)
        else:
            rlayer_ds = SamTestRasterDataset(
                root=rlayer_dir, crs=crs.toWkt(), res=self.res, bands=input_bands, cache=False)
        # \n raster_ds crs: {str(CRS(rlayer_ds.crs))}, \
        feedback.pushInfo(
            f'\n RasterDataset info: \
            \n filename_glob: {rlayer_ds.filename_glob}, \
            \n all bands: {rlayer_ds.all_bands}, \
            \n input bands: {rlayer_ds.bands}, \
            \n resolution: {rlayer_ds.res}, \
            \n bounds: {rlayer_ds.index.bounds}, \
            \n num: {len(rlayer_ds.index)}\n')

        # feedback.pushInfo(f'raster dataset crs: {rlayer_ds.crs}')

        extent_bbox = BoundingBox(minx=extent.xMinimum(), maxx=extent.xMaximum(), miny=extent.yMinimum(), maxy=extent.yMaximum(),
                                  mint=rlayer_ds.index.bounds[4], maxt=rlayer_ds.index.bounds[5])

        self.sam_model = self.initialize_sam(
            model_type=model_type, sam_ckpt_path=ckpt_path)
        
        feedback.pushInfo(f"timm version : {timm.__version__}")
        feedback.pushInfo(f'QGIS Python Path : {sys.executable}')
        if 'CONDA_DEFAULT_ENV' in os.environ:
            conda_environment = os.environ['CONDA_DEFAULT_ENV']
            feedback.pushInfo("Conda Environment:", conda_environment)
        else:
            feedback.pushInfo("Not running in a Conda environment.")
        #self.sam_model = sam_first_layer_with_nchan(self.sam_model, len(input_bands))
        timm_model = timm.create_model(
                'samvit_large_patch16.sa1b',
                pretrained=True,
                in_chans=len(input_bands)
                )
        
        self.sam_model.image_encoder = timm_model
        #One can change it freely, with the condition that it should always be bigger than the stride
        self.sam_model.image_encoder.img_size = 1024
        
        self.sam_model.pixel_mean = torch.Tensor(MEANS)
        self.sam_model.pixel_std = torch.Tensor(SDS)
        feedback.pushInfo(f'{self.sam_model}')

        ds_sampler = SamTestGridGeoSampler(
            rlayer_ds, size=self.sam_model.image_encoder.img_size, stride=stride, roi=extent_bbox, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(ds_sampler) == 0:
            self.load_feature = False
            feedback.pushWarning(
                f'\n !!!No available patch sample inside the chosen extent!!! \n')
            # return {'Input layer dir': rlayer_dir, 'Sample num': len(ds_sampler.res),
            #         'Sample size': len(ds_sampler.size), 'Sample stride': len(ds_sampler.stride)}

        feedback.pushInfo(
            f'SAM model initialized. \n \
              SAM model type:  {model_type}')
        if torch.cuda.is_available() and self.use_gpu:
            feedback.pushInfo(
                f'Device type: {self.sam_model.device} on {torch.cuda.get_device_name(self.sam_model.device)}')
        else:
            batch_size = 1
            feedback.pushInfo(
                f'Device type: {self.sam_model.device}')

        feedback.pushInfo(
            f'Patch size: {ds_sampler.patch_size} \n \
            Batch size: {batch_size}')
        ds_dataloader = DataLoader(
            rlayer_ds, batch_size=batch_size, sampler=ds_sampler, collate_fn=stack_samples)

        feedback.pushInfo(f'Patch sample num: {len(ds_sampler)}')
        feedback.pushInfo(f'Total batch num: {len(ds_dataloader)}\n \
                          ----------------------------------------------------')

        elapsed_time_list = []
        total = 100 / len(ds_dataloader) if len(ds_dataloader) else 0
        #start of the core of the algorithm
        #initialization of the bboxes list
        bboxes = []
        
        for current, batch in enumerate(ds_dataloader):
            start_time = time.time()
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                self.load_feature = False
                feedback.pushWarning(
                    self.tr("\n !!!Processing is canceled by user!!! \n"))
                break
            feedback.pushInfo(f'Batch no. {current+1} loaded')
            feedback.pushInfo(f'img_shape: ' + str(batch['img_shape'][0]))
            feedback.pushInfo('patch_size: ' + str(batch['image'].shape))

            self.batch_input = self.rescale_img(
                batch_input=batch['image'], range_value=range_value)

            if not self.get_sam_feature(self.batch_input, feedback):
                self.load_feature = False
                break
            #To have the bboxes list
            for sample in unbind_samples(batch):
                bboxes.append(sample['bbox'])

            end_time = time.time()
            # get the execution time of sam predictor, ms
            elapsed_time = (end_time - start_time)
            elapsed_time_list.append(elapsed_time)
            time_spent = sum(elapsed_time_list)
            time_remain = (time_spent / (current + 1)) * \
                (len(ds_dataloader) - current - 1)
            feedback.pushInfo('feature_shape:' + str(self.features.shape))

            # TODO: show gpu usage info
            # if torch.cuda.is_available() and self.use_gpu:
            #     gpu_mem_used = torch.cuda.max_memory_reserved(self.sam_model.device) / (1024 ** 3)
            #     # gpu_mem_free = torch.cuda.mem_get_info(self.sam_model.device)[0] / (1024 ** 3)
            #     gpu_mem_total = torch.cuda.mem_get_info(self.sam_model.device)[1] / (1024 ** 3)
            #     feedback.pushInfo(
            #         f'GPU memory usage: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB')
            #     feedback.pushInfo(str(torch.cuda.memory_summary(self.sam_model.device)))

            feedback.pushInfo(
                f"SAM encoder executed with {elapsed_time:.3f}s \n \
                  Time spent: {time_spent:.3f}s")
            if time_remain <= 60:
                feedback.pushInfo(f"Estimated time remaining: {time_remain:.3f}s \n \
                                  ----------------------------------------------------")
            else:
                time_remain_m, time_remain_s = divmod(int(time_remain), 60)
                time_remain_h, time_remain_m = divmod(time_remain_m, 60)
                feedback.pushInfo(f"Estimated time remaining: {time_remain_h:d}h:{time_remain_m:02d}m:{time_remain_s:02d}s \n \
                                  ----------------------------------------------------")

            self.feature_dir = self.save_sam_feature(
                output_dir, batch, self.features, extent_bbox, model_type)

            # Update the progress bar
            feedback.setProgress(int((current+1) * total))
        if(self.FEAT_OPTION == True) :
            
            feat_array = np.stack(list_features, axis = 0)
            Nx = len(bboxes)
            for i in range(1, len(bboxes)):
                if bboxes[i][0] < bboxes[i - 1][0]:
                    Nx = i
                    break
            Ny = int(len(bboxes) / Nx)
        
            """""
            feedback.pushInfo(f"length of Nx : {Nx}")
            feedback.pushInfo(f"length of Ny : {Ny}")
        
            image_shape = feat_array.shape[2:]  # Shape of each tensor
            feedback.pushInfo(f"Dim de image_shape : {len(image_shape)}")
    
            channels, h, w = image_shape
            feedback.pushInfo(f"nbr de channels : {channels}")
            feedback.pushInfo(f"Hauteur : {h}")
            feedback.pushInfo(f"Largeur : {w}")
    
            reconstructed_height = h * Ny
            reconstructed_width = w * Nx
            feedback.pushInfo(f"hauteur de l'image reconstruite : {reconstructed_height}")
            feedback.pushInfo(f"largeur de l'image reconstruite : {reconstructed_width}")
            """""
            
            macro_img= reconstruct_img_feat(feat_array, Nx, Ny)
            #tifffile.imsave('C:/Users/pierr/OneDrive/Documents/Administratif/Thaïlande/testfeatoption.tiff', macro_img)
            patch_size = 16 #depends on the kind of ViT you're using
            
            pca = PCA(3) # take 3 principal components.
            pca_img = pca.fit_transform(macro_img.reshape(-1, macro_img.shape[-1]))
            macro_img = pca_img.reshape((macro_img.shape[0], macro_img.shape[1],-1))
            cwd = Path(__file__).parent.parent.absolute()
            
            output_directory = os.path.join(cwd, 'rasters')
            output_file_base = 'testfeat.tiff'
            output_file = os.path.join(output_directory, output_file_base)


            if os.path.exists(output_file):
                i = 1
                while True:
                    modified_output_file = os.path.join(output_directory, f"{output_file_base.split('.')[0]}_{i}.tiff")
                    if not os.path.exists(modified_output_file):
                        output_file = modified_output_file
                        break
                    i += 1
            
            array_to_geotiff(
               array=macro_img,
               top_left_corner_coords= (bboxes[0].minx, bboxes[-1].maxy),
               pixel_height= rlayer.rasterUnitsPerPixelX()*patch_size,
               pixel_width=rlayer.rasterUnitsPerPixelY()*patch_size,
               crs = rlayer.crs().authid(),
               #output_file='C:/Users/pierr/OneDrive/Documents/Administratif/Thaïlande/testfeatoption.tiff',
               #output_file = os.path.join(cwd,'rasters','testfeat.tiff'),
               output_file = output_file,
            )
        
        

        return {"Output feature path": self.feature_dir, 'Patch samples saved': self.iPatch, 'Feature folder loaded': self.load_feature}

    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        if torch.cuda.is_available() and self.use_gpu:
            if hasattr(self, 'sam_model'):
                del self.sam_model
            if hasattr(self, 'batch_input'):
                del self.batch_input
            torch.cuda.empty_cache()
        if self.load_feature and self.feature_dir:
            self.load_feature = self.load_feature_dir(feedback=feedback)
        return {"Output feature path": self.feature_dir, 'Patch samples saved': self.iPatch, 'Feature folder loaded': self.load_feature}

    def load_feature_dir(self, feedback: QgsProcessingFeedback) -> bool:
        sam_tool_action: QAction = iface.mainWindow().findChild(QAction,
                                                                "mActionGeoSamTool")
        if sam_tool_action:
            sam_tool_action.trigger()
            start_time = time.time()
            while True:
                if feedback.isCanceled():
                    feedback.pushInfo(
                        self.tr("Loading feature is canceled by user."))
                    return False
                sam_tool_widget: QgsDockWidget = iface.mainWindow().findChild(QDockWidget, 'GeoSAM')
                current_time = time.time()
                elapsed_time = (current_time - start_time) * 1000
                if sam_tool_widget:
                    load_feature_widget: QgsFileWidget = sam_tool_widget.QgsFile_feature
                    load_feature_widget.setFilePath(self.feature_dir)
                    sam_tool_widget.pushButton_load_feature.click()  # try sender
                    feedback.pushInfo(
                        f'\n GeoSAM widget found and features loaded in {elapsed_time:.3f} ms \n')
                    return True
                # try 3 seconds
                if elapsed_time > 3000:
                    feedback.pushInfo(
                        f'\n GeoSAM widget not found {elapsed_time:.3f} ms \n')
                    return False
        else:
            feedback.pushInfo('\n GeoSAM tool action not found. \n')
            return False

    def initialize_sam(self, model_type: str, sam_ckpt_path: str) -> Sam:
        sam_model = sam_model_registry[model_type](
            checkpoint=sam_ckpt_path)
        if torch.cuda.is_available() and self.use_gpu:
            if self.cuda_id + 1 > torch.cuda.device_count():
                self.cuda_id = torch.cuda.device_count() - 1
            cuda_device = f'cuda:{self.cuda_id}'
            sam_model.to(device=cuda_device)
        return sam_model

    @torch.no_grad()
    def get_sam_feature(self, batch_input: Tensor, feedback: QgsProcessingFeedback) -> bool:
        # TODO: if the input image are all zero(batch_input.any()), directly return features with all zero and give a message
        # should know the shape of the feature in advance
        batch_input = batch_input.to(device=self.sam_model.device)
        feedback.pushInfo(f'{self.sam_model.pixel_mean.shape}')
        feedback.pushInfo(f'QGIS Python Path : {sys.executable}')
        if len(self.sam_model.pixel_mean.shape) == 1:  
            self.sam_model.pixel_mean = self.sam_model.pixel_mean.unsqueeze(1).unsqueeze(2)
            self.sam_model.pixel_std = self.sam_model.pixel_std.unsqueeze(1).unsqueeze(2)
        batch_input = ((batch_input - self.sam_model.pixel_mean) /
                       self.sam_model.pixel_std)
        # batch_input = sam_model.preprocess(batch_input)
        try:
            features = self.sam_model.image_encoder.forward_features(batch_input)
            
            feedback.pushInfo(f'Dimension des features : {features.size()}')
            list_features.append(features)
            feedback.pushInfo(f'using timm encoder')
            feedback.pushInfo(f'Nbr de features enregistrées : {len(list_features)}')
        except RuntimeError as inst:
            # torch.cuda.OutOfMemoryError
            if 'CUDA out of memory' in inst.args[0]:
                feedback.pushWarning(
                    "\n !!!CUDA out of memory, try to choose a smaller batch size or smaller version of SAM model.!!!")
                feedback.pushWarning(
                    f'Error type: {type(inst).__name__}, context: {inst} \n'
                )
            # raise QgsProcessingException(
            #     f'Error type: {type(inst).__name__}, context: {inst}')
            return False
        except Exception as err:
            raise QgsProcessingException(f"Unexpected {err=}, {type(err)=}")
        # batch_input = batch_input.to(device='cpu')
        # torch.cuda.empty_cache()
        except:
            features = self.sam_model.image_encoder(batch_input)
        self.features = features.cpu().numpy()
        return True

    def rescale_img(self, batch_input: Tensor, range_value: List[float]) -> Tensor:
        'rescale input image to [0,255]'
        range_min = range_value[0]
        range_max = range_value[1]
        batch_output = (batch_input - range_min)*255/(range_max - range_min)
        return batch_output

    def save_sam_feature(
        self,
        export_dir_str: str,
        data_batch: Tensor,
        feature: np.ndarray,
        extent: BoundingBox,
        model_type: str = "vit_h"
    ) -> str:
        export_dir = Path(export_dir_str)
        # iterate over batch_size dimension
        for idx in range(feature.shape[-4]):
            band_num = feature.shape[-3]
            height = feature.shape[-2]
            width = feature.shape[-1]
            bbox = data_batch['bbox'][idx]
            rio_transform = rasterio.transform.from_bounds(
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)  # west, south, east, north, width, height
            filepath = Path(data_batch['path'][idx])
            bbox_list = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
            bbox_str = '_'.join(map("{:.6f}".format, bbox_list))
            extent_list = [extent.minx, extent.miny, extent.maxx, extent.maxy]
            extent_str = '_'.join(
                map("{:.6f}".format, extent_list)) + f"_res_{self.res:.6f}"
            #  Unicode-objects must be encoded before hashing with hashlib and
            #  because strings in Python 3 are Unicode by default (unlike Python 2),
            #  you'll need to encode the string using the .encode method.
            bbox_hash = hashlib.sha256(bbox_str.encode("utf-8")).hexdigest()
            extent_hash = hashlib.sha256(
                extent_str.encode("utf-8")).hexdigest()

            bands_str = '_'.join([str(band) for band in self.selected_bands])
            export_dir_sub = (export_dir / filepath.stem /
                              f"sam_feat_{model_type}_bands_{bands_str}_{extent_hash[0:16]}")
            export_dir_sub.mkdir(parents=True, exist_ok=True)
            feature_tiff = (export_dir_sub /
                            f"sam_feat_{model_type}_{bbox_hash}.tif")
            feature_csv = (export_dir_sub / f"{export_dir_sub.name}.csv")
            with rasterio.open(
                    feature_tiff,
                    mode="w",
                    driver="GTiff",
                    height=height, width=width,
                    count=band_num,
                    dtype='float32',
                    crs=data_batch['crs'][idx],
                    transform=rio_transform
            ) as feature_dataset:
                # index start from 1, feature[idx, :, :, :] = feature[idx, ...], later is faster
                #Peut être lui passer list_features[:, idx, :, :, :]
                feature_dataset.write(feature[idx, ...], range(1, band_num+1))
                # pr_mask_dataset.set_band_description(1, '')
                tags = {
                    "img_shape": data_batch["img_shape"][idx],
                    "input_shape": data_batch["input_shape"][idx],
                    "model_type": model_type,
                }
                feature_dataset.update_tags(**tags)
                feature_res = feature_dataset.res[0]
                feature_crs = feature_dataset.crs
            #In case of need to export each sub images in a different tiff :
            #feature_tiff_path = str(feature_tiff)
            #command = f'qgis {feature_tiff_path}'
            #subprocess.Popen(command, shell=True)

            index_df = pd.DataFrame(columns=['minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt',
                                             'filepath',
                                             'crs', 'res'],
                                    index=[self.iPatch])
            index_df['filepath'] = [feature_tiff.name]
            index_df['minx'] = [bbox.minx]
            index_df['maxx'] = [bbox.maxx]
            index_df['miny'] = [bbox.miny]
            index_df['maxy'] = [bbox.maxy]
            index_df['mint'] = [bbox.mint]
            index_df['maxt'] = [bbox.maxt]
            index_df['crs'] = [str(feature_crs)]
            index_df['res'] = [self.res]
            index_df['model_type'] = [model_type]
            # append data frame to CSV file, index=False
            index_df.to_csv(feature_csv, mode='a',
                            header=not feature_csv.exists(), index=True)
            self.iPatch += 1

        return str(export_dir_sub)

    def estimate_utm_crs(self, extent: QgsRectangle):
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=extent.xMinimum(),
                south_lat_degree=extent.yMinimum(),
                east_lon_degree=extent.xMaximum(),
                north_lat_degree=extent.yMaximum(),
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        utm_crs = QgsCoordinateReferenceSystem(str(utm_crs))
        return utm_crs

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SamProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'geo_sam_encoder'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Geo-SAM Image Encoder')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return ''

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        file = encoder_help
        if not os.path.exists(file):
            return self.tr("Generate image features using SAM image encoder.")
        with open(file) as help_file:
            help_str = help_file.read()
        return help_str
        # return self.tr("Generate image features using SAM image encoder.")

    def icon(self):
        return QIcon_EncoderTool
