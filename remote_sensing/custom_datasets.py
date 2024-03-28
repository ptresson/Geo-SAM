from torchgeo.datasets import RasterDataset, stack_samples
from torch.utils.data import DataLoader
# from utils import stack_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

class Ortho_1959(RasterDataset):
    filename_glob = "1959.tif"
    is_image = True
    separate_files = True
    all_bands = ["..."]

# class Covar_maille(RasterDataset):
#     filename_glob = "*.tif"
#     is_image = True
#     # separate_files = False
#     separate_files = True
#     # all_bands = ["..."]
#     all_bands = [
#                    # "bi", 
#                  "catchment_area",
#                  "dem_zone_reproj",
#     #              "ls_factor", 
#     #              "ndsi",
#     #              "ndvi",
#     #              "slope",
#     #              "TWI",
#                  ]

class BI(RasterDataset):
    filename_glob = "bi.tif"
    is_image = True
    separate_files = False

class CA(RasterDataset):
    filename_glob = "catchment_area.tif"
    is_image = True
    separate_files = False

class DEM(RasterDataset):
    filename_glob = "dem_zone_reproj.tif"
    is_image = True
    separate_files = False

class LS(RasterDataset):
    filename_glob = "ls_factor.tif"
    is_image = True
    separate_files = False

class NDSI(RasterDataset):
    filename_glob = "ndsi.tif"
    is_image = True
    separate_files = False

class NDVI(RasterDataset):
    filename_glob = "ndvi.tif"
    is_image = True
    separate_files = False

class Slope(RasterDataset):
    filename_glob = "slope.tif"
    is_image = True
    separate_files = False

class TWI_maille(RasterDataset):
    filename_glob = "TWI.tif"
    is_image = True
    separate_files = False

class SAR_maille(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    separate_files = False


class Spot_maille(RasterDataset):
    filename_glob = "mars2019.tif"
    is_image = True
    separate_files = False
    all_bands = [0,1,2]

class Spot_maille_alpha(RasterDataset):
    filename_glob = "mars2019.tif"
    is_image = True
    separate_files = False
    # all_bands = [0,1,2]

if __name__ == "__main__":
    
    bi = BI('/home/ptresson/maille/ADMaille/')
    catchment_area = CA('/home/ptresson/maille/ADMaille/')
    dem_zone_reproj = DEM('/home/ptresson/maille/ADMaille/')
    ls_factor = LS('/home/ptresson/maille/ADMaille/')
    ndsi = NDSI('/home/ptresson/maille/ADMaille/')
    ndvi = NDVI('/home/ptresson/maille/ADMaille/')
    slope = Slope('/home/ptresson/maille/ADMaille/')
    TWI = TWI_maille('/home/ptresson/maille/ADMaille/')

    spot = Spot_maille('/home/ptresson/maille/')
    sar = SAR_maille('/home/ptresson/maille/ADMaille/SAR/')

    dataset = spot
    dataset = dataset & sar

    dataset = dataset & bi
    dataset = dataset & catchment_area
    dataset = dataset & dem_zone_reproj
    dataset = dataset & ls_factor
    dataset = dataset & ndsi
    dataset = dataset & ndvi
    dataset = dataset & slope
    dataset = dataset & TWI


    sampler = RandomGeoSampler(dataset, size = 224, length = 10)
    dataloader = DataLoader(dataset, 10, sampler=sampler, collate_fn=stack_samples)

    for batch in dataloader:
        # print('pouet')
        print(batch['image'].shape)
