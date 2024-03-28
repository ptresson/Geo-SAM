# import shapefile as shp  # Requires the pyshp package
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import adjust_band, show, reshape_as_image, reshape_as_raster
import rasterio.mask

# def masked_raster(input_file, raster_file):
#     # Create a masked version of the input raster where pixels falling within one of the fields are set to `1` and pixels outside the fields are set to `0`
   
#     geoms = [input_file.geometry]
#     # load the raster, mask it by the polygon and crop it
#     with rasterio.open(raster_file) as src:
#         out_img, out_transform = rasterio.mask(src, geoms, invert=True)
#     out_meta = src.meta.copy()

#     # save the resulting raster  
#     out_meta.update({"driver": "GTiff",
#     "height": out_image.shape[1],
#     "width": out_image.shape[2],
#     "transform": out_transform})

    
#     return out_img

# def reproject_raster(raster_file, dst_crs):
#     # Reproject the input raster to the provided CRS
    
#     with rasterio.open("masked.tif", "w", **out_meta) as dst:
#       dst.write(out_image)


#     dst = src
    
#     return dst

raster = rasterio.open("./out/hegra_raster_cut3.tif")
df=gpd.read_file("./out/clusters/kmeans8.shp")
gdf=gpd.read_file("./hegra.gpkg", layer="habitat_hegra_paul")
df = df.to_crs(raster.crs)
gdf = gdf.to_crs(raster.crs)

out_image,out_transform=rasterio.mask.mask(raster,[gdf.geometry.unary_union],crop=True)
print(out_image.shape)

# with rasterio.open(original_image) as ds :
#     transform = ds.meta.copy()['transform']
#     transform = list(transform)
#     # 0 and 4th element of transform correspond to horizontal and vertical pixel size
#     transform[0] = transform[0] * pixel_scale
#     transform[4] = transform[4] * pixel_scale
#     crs = ds.meta.copy()['crs']
#     ds.close()
# metadata = {'driver': 'GTiff', 
#             'dtype':dtype, 
#             'nodata': None, 
#             'width': numpy_image.shape[1], 
#             'height': numpy_image.shape[0], 
#             'count': n_channels, 
#             'crs': crs, 
#             'transform':Affine(
#                 *transform[0:6]
#                 )}
            
# with rasterio.open('out/test.tif', "w", **metadata) as ds:
#     ds.write(np.transpose(numpy_image, (2, 0, 1)))

fig,ax = plt.subplots()


# xlim = ([extent.total_bounds[0],  extent.total_bounds[2]])
# ylim = ([extent.total_bounds[1],  extent.total_bounds[3]])

# ax.set_xlim(xlim)
# ax.set_ylim(ylim)


# rasterio.plot.show( 
#                    # band1,
#                    # adjust_band(raster.read([1,2,3])),
#                    out_image,
#                    # (out_image,1),
#                    # (raster,1),
#                    ax=ax,
#                    # cmap = 'Reds_r',
#                    # alpha = 0.3,
#                    zorder=4,
#                    vmin = 1
#                    );


gdf.boundary.plot(
        ax=ax,
        edgecolor = 'white', 
        linewidth = 0.5,
        zorder=1000
        );

df.plot(
        ax=ax,
        cmap='Set2',
        column='label',
        zorder=0
        );

ax.set_axis_off();
plt.tight_layout();

fig.savefig("out/test.png")
