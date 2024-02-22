import sys
import os
import rasterio
from rasterio.transform import from_origin

def georeference_tiff(input_filename, output_filename):
    # Open the input raster file
    with rasterio.open(input_filename) as src:
        # Get raster dimensions
        width = src.width
        height = src.height
        
        # Set an arbitrary transformation
        transform = from_origin(0, 0, 1, 1)  # Adjust the pixel size as needed
        
        # Create the output raster profile with an arbitrary CRS (coordinate reference system)
        profile = src.profile
        profile.update(transform=transform, crs='EPSG:4326')  # EPSG:4326 is WGS 84
        
        # Modify output filename with "geo_" prefix
        output_filename = output_filename.replace(os.path.basename(output_filename), "geo_" + os.path.basename(output_filename))
        
        # Write to the output raster file
        with rasterio.open(output_filename, 'w', **profile) as dst:
            data = src.read()
            dst.write(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input_file1.tif input_file2.tif ...")
        sys.exit(1)

    input_files = sys.argv[1:]
    for input_file in input_files:
        output_file = "geo_" + os.path.basename(input_file)
        georeference_tiff(input_file, output_file)
        print(f"Georeferenced TIFF saved as {output_file}")
