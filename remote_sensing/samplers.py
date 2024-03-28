from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import numpy as np
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips
import geopandas as gpd
import abc
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
from rtree.index import Index, Property


class GeoSampler(Sampler[BoundingBox], abc.ABC):

    def __init__(self, dataset: GeoDataset, rois: Optional[BoundingBox] = None) -> None:
        if rois is None:
            self.index = dataset.index
            rois = [BoundingBox(*self.index.bounds)]
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            for roi in rois: 
                hits = dataset.index.intersection(tuple(roi), objects=True)
                for hit in hits:
                    bbox = BoundingBox(*hit.bounds) & roi
                    self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.rois = rois

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        pass

class GridShpGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for roi in self.rois :
            for hit in self.index.intersection(tuple(roi), objects=True):
                """
                bounds = BoundingBox(*hit.bounds)
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                """
                self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]
                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                                and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                        self.length+=1

    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        k=0
        for hit in self.hits:
            k+=1
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt

            # For each row...
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]

                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    #if the bounding box intersects with a roi, we keep it
                    if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                                and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                        yield bounding_box  

    def __len__(self) -> int:
        return self.length

"""
# maybe check if this one works better if
# there is any problems in the future ?
class GridShpGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for roi in self.rois :
            for hit in self.index.intersection(tuple(roi), objects=True):

                self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]
                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                                and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                        self.length+=1

    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        k=0
        for hit in self.hits:
            k+=1
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt

            # For each row...
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]

                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    #if the bounding box intersects with a roi, we keep it
                    if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                                and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                        yield bounding_box    

    def __len__(self) -> int:
        return self.length
"""    

class RandomOutsideRoiGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: Optional[int],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)

        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        for roi in self.rois:
            for hit in self.index.intersection(tuple(roi), objects=True):
                bounds = BoundingBox(*hit.bounds)
                """
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                """
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1
        
    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(len(self)):
            val = False
            while not val :
                val = True
                bounding_box = get_random_bounding_box(self.global_bounds, self.size, self.res)
                for hit in self.hits:
                    bounds = BoundingBox(*hit.bounds)
                    #if the random bounding box intersects with a roi, we keep it                    
                    # if  (bounding_box.maxx<bounds.minx or bounding_box.minx>bounds.maxx) \
                    #             and (bounding_box.maxy<bounds.miny or bounding_box.miny>bounds.maxy):         
                    if  (bounds.minx<bounding_box.minx<bounds.maxx or bounds.minx<bounding_box.maxx<bounds.maxx) \
                                and (bounds.miny<bounding_box.miny<bounds.maxy or bounds.miny<bounding_box.maxy<bounds.maxy):         
                        val = False
                        break  

            yield bounding_box

    def __len__(self) -> int:
        return self.length


class RandomOutsideRoisGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: Optional[int],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)

        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        for roi in self.rois:
            for hit in self.index.intersection(tuple(roi), objects=True):
                bounds = BoundingBox(*hit.bounds)
                """
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                """
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1
        
    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(len(self)):
            val = False
            while not val :
                val = True
                bounding_box = get_random_bounding_box(self.global_bounds, self.size, self.res)
                for hit in self.hits:
                    bounds = BoundingBox(*hit.bounds)
                    #if the random bounding box intersects with a roi, we keep it                    
                    if  (bounding_box.minx<bounds.minx<bounding_box.maxx or bounding_box.minx<bounds.maxx<bounding_box.maxx) \
                                and (bounding_box.miny<bounds.miny<bounding_box.maxy or bounding_box.miny<bounds.maxy<bounding_box.maxy):         
                        val = False
                        break  

            yield bounding_box

    def __len__(self) -> int:
        return self.length



class CenterRoiGeoSampler(GeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)
        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.hits = []
        for roi in self.rois :
            for hit in self.index.intersection(tuple(roi), objects=True):
                if list(roi)==hit.bounds:
                    self.hits.append(hit)

        self.length = len(rois)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        for hit in self.hits:
            roi_bounds = BoundingBox(*hit.bounds)
            
            margin = int(self.size/2)
            Cx, Cy =  int((roi_bounds.minx+roi_bounds.maxx)/2), int((roi_bounds.maxy+roi_bounds.miny)/2)
            bounding_box = BoundingBox(Cx-margin, Cx+margin+self.size%2, Cy-margin, Cy+margin+self.size%2)

            yield bounding_box  

    def __len__(self) -> int:
        return self.length

class GridOutsideGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for roi in self.rois :
            for hit in self.index.intersection(tuple(roi), objects=True):
                """
                bounds = BoundingBox(*hit.bounds)
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                """
                self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]
                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    # if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                    #             and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                    if not ((roi_bounds.minx<bounding_box.minx<roi_bounds.maxx or roi_bounds.minx<bounding_box.maxx<roi_bounds.maxx) \
                                and (roi_bounds.miny<bounding_box.miny<roi_bounds.maxy or roi_bounds.miny<bounding_box.maxy<roi_bounds.maxy)):         
                        self.length+=1

    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        k=0
        for hit in self.hits:
            k+=1
            roi_bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(self.global_bounds, self.size, self.stride)
            mint = self.global_bounds.mint
            maxt = self.global_bounds.maxt

            # For each row...
            for i in range(rows):
                miny = self.global_bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > self.global_bounds.maxy:
                    maxy = self.global_bounds.maxy
                    miny = self.global_bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = self.global_bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > self.global_bounds.maxx:
                        maxx = self.global_bounds.maxx
                        minx = self.global_bounds.maxx - self.size[1]

                    bounding_box = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    #if the bounding box intersects with a roi, we keep it
                    # if (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                    #             and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):
                    # if  (bounding_box.minx<roi_bounds.minx<bounding_box.maxx or bounding_box.minx<roi_bounds.maxx<bounding_box.maxx) \
                    #             and (bounding_box.miny<roi_bounds.miny<bounding_box.maxy or bounding_box.miny<roi_bounds.maxy<bounding_box.maxy):         
                    # if  (roi_bounds.minx<bounding_box.minx<roi_bounds.maxx or roi_bounds.minx<bounding_box.maxx<roi_bounds.maxx) \
                    #             and (roi_bounds.miny<bounding_box.miny<roi_bounds.maxy or roi_bounds.miny<bounding_box.maxy<roi_bounds.maxy):         
                    if not ((roi_bounds.minx<bounding_box.minx<roi_bounds.maxx or roi_bounds.minx<bounding_box.maxx<roi_bounds.maxx) \
                                and (roi_bounds.miny<bounding_box.miny<roi_bounds.maxy or roi_bounds.miny<bounding_box.maxy<roi_bounds.maxy)):         
                        yield bounding_box    

    def __len__(self) -> int:
        return self.length

class RoiGeoSampler(GeoSampler):
    """
    !!! Only returns Bounding Boxes that have the same size as the different rois !!!
    """
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        rois: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, rois)
        self.size = _to_tuple(size)
        self.global_bounds = BoundingBox(*dataset.index.bounds)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.hits = []
        for roi in self.rois :
            for hit in self.index.intersection(tuple(roi), objects=True):
                if list(roi)==hit.bounds:
                    self.hits.append(hit)

        self.length = len(rois)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        for hit in self.hits:
            yield BoundingBox(*hit.bounds) 

    def __len__(self) -> int:
        return self.length
