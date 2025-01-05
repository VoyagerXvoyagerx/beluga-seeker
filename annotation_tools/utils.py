import os
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import json
from datetime import datetime
import re
from rasterio.transform import rowcol


def get_strip_id(tile):
    match = re.search(r'P00\d', tile)
    if match:
        return match.group(0)
    else:
        print("No strip ID found in the file name.")


def get_tile_path(root_dir, tile):
    """
    Returns the file path for a specified tile based on the given root directory.

    If the root directory contains 'P00', the tile is assumed to be directly under this directory.
    Otherwise, the function calculates the `strip_id` from the tile name and returns the path based
    on this `strip_id`.
    
    e.g. directory tree for Churchill 22 data which consists of 2 strips:
        E:\2022\Churchill_050166593010 & 050169967010>
        +---050169967010_01_P001_PAN
        |       22JUL31174054-P3DS_R1C1-050169967010_01_P001.TIF
        |       22JUL31174054-P3DS_R3C2-050169967010_01_P001.TIF
        |       ...
        +---050169967010_01_P002_PAN
        |       22JUL31174035-P3DS_R2C1-050169967010_01_P002.TIF
        |       22JUL31174035-P3DS_R2C2-050169967010_01_P002.TIF
        |       ...
        \---GIS_FILES
        
    e.g. directory tree for Clearwater 2022 data which consist of only 1 strip:
        E:\2022\Clearwater&Kangilo_015379185020_01>
        +---015379185020_01_P001_PAN_MOS
        |       22AUG01161709-P3DM_R01C1-015379185020_01_P001.TIF
        |       22AUG01161709-P3DM_R01C2-015379185020_01_P001.TIF
        |       ...
        +---GIS_FILES
        |       015379185020_01_ORDER_SHAPE.shp
        |       ...
        \---WS_Clearwater&Kangilo_015379185020_01_annotations
            \---Clearwater&Kangilo_015379185020_01.shp

    Args:
        root_dir (str): The root directory path. If it contains 'P00', the tile is located directly in this directory else the tile is located in a subdirectory named after the `strip_id`.
        tile (str): The name of the tile (excluding the file extension).

    Returns:
        str: The full file path of the tif file.
    """
    if 'P00' in root_dir:   # e.g. D:\Whale_Data\Clearwater&Kangilo_015379185020_01\015379185020_01_P001_PAN_MOS
        return os.path.join(root_dir, f"{tile}.tif")
    else:                   # e.g. 'E:\2022\Churchill_050166593010 & 050169967010\050169967010_01\050169967010_01_'
        strip_id = get_strip_id(tile)
        return os.path.join(f"{root_dir}{strip_id}_PAN", f"{tile}.tif")


# Function to process tiles and generate metadata
def crop_tiles(tiles_with_targets, pan_dir, whale_gdf, dataSource, crop_size=512, output_dir='output_crops', metadata_path=None, append=False):
    # Initialize JSON data structure
    if append and metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "type": "AI_EOTrainingDataset",
            "name": "Dataset for Beluga Whale Detection",
            "description": "Dataset for Beluga Whale Detection",
            "license": "",
            "version": "0.1.0",
            "createTime": datetime.now().strftime("%Y-%m-%d"),
            "providers": ["University of Chinese Academy of Sciences"],
            "classes": ["certain whale", "uncertain whale", "harp seal"],
            "numberOfClasses": 3,
            "bands": ["panchromatic"],
            "amountOfTrainingTiles": 0,
            "amountOfCroppedData": 0,
            "amountOfTrainingLabels": 0,
            "tiles": []
        }

    for tile in tiles_with_targets:
        pan_path = get_tile_path(pan_dir, tile)
        if os.path.exists(pan_path):
            try:
                with rasterio.open(pan_path) as src:
                    print(f"Tile: {tile}  Size: {src.width} x {src.height}")
                    print(f"Bounds: {src.bounds}")

                    # Initialize tile metadata
                    tile_metadata = {
                        "tileURL": f"{tile}.tif",
                        "dataSources": [dataSource],
                        "imagerySize": [src.width, src.height],
                        "cropSize": [crop_size, crop_size],
                        "numberOfCrops": 0,
                        "numberOfLabels": 0,
                        "windows": []
                    }

                    # Get whale positions in this tile
                    whale_positions = whale_gdf[whale_gdf['Photo-ID'] == tile]

                    # Get the CRS of the raster and the whale positions
                    raster_crs = src.crs
                    whale_crs = whale_gdf.crs

                    # Create a transformer to convert whale positions to the raster CRS
                    transformer = Transformer.from_crs(whale_crs, raster_crs, always_xy=True)

                    crop_info = {}

                    for _, whale in whale_positions.iterrows():
                        geom = whale.geometry
                        if not geom.is_empty:
                            # Convert whale geometry to raster coordinates
                            x, y = transformer.transform(geom.x, geom.y)
                            row, col = src.index(x, y)
                            print(row, col)
                            # Determine crop window indices
                            crop_row = row // crop_size
                            crop_col = col // crop_size
                            window = Window(crop_col * crop_size, crop_row * crop_size, crop_size, crop_size)

                            # Read the crop window
                            crop = src.read(1, window=window)

                            # Save crop as a new TIFF file
                            crop_filename = f"{tile}_{crop_row}_{crop_col}.tif"
                            crop_path = os.path.join(output_dir, crop_filename)
                            os.makedirs(os.path.dirname(crop_path), exist_ok=True)

                            with rasterio.open(crop_path, "w", driver="GTiff", height=crop.shape[0], width=crop.shape[1], count=1, dtype=crop.dtype, crs=raster_crs, transform=src.window_transform(window)) as dest:
                                dest.write(crop, 1)

                            # Calculate the point's position in the crop window
                            point_row = row - crop_row * crop_size
                            point_col = col - crop_col * crop_size
                            # Update crop information
                            if (crop_row, crop_col) not in crop_info:
                                crop_info[(crop_row, crop_col)] = {
                                    "cropID": [crop_row, crop_col],
                                    "dataURL": crop_filename,
                                    "numberOfLabels": 0,
                                    "labels": []
                                }

                            crop_info[(crop_row, crop_col)]["labels"].append({
                                "class": whale["Species"],
                                "originID": whale["ID"],
                                "pointIndex": [point_row, point_col]
                            })

                            crop_info[(crop_row, crop_col)]["numberOfLabels"] += 1

                    # Add crop info to tile metadata
                    for key, value in crop_info.items():
                        tile_metadata["windows"].append(value)

                    tile_metadata["numberOfCrops"] += len(crop_info)
                    tile_metadata["numberOfLabels"] += sum([info["numberOfLabels"] for info in crop_info.values()])

                    # Add tile metadata to JSON data
                    json_data["tiles"].append(tile_metadata)

                    json_data["amountOfTrainingTiles"] += 1
                    json_data["amountOfCroppedData"] += tile_metadata["numberOfCrops"]
                    json_data["amountOfTrainingLabels"] += tile_metadata["numberOfLabels"]
            except rasterio.errors.RasterioIOError as e:
                print(f"Error reading {pan_path}: {e}")

    # Save JSON data to a file
    json_output_path = os.path.join(output_dir, "metadata.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Saved metadata to {json_output_path}")
    
    
def crop_aerial_tiles(tiles_with_targets, pan_dir, whale_gdf, dataSource, crop_size=512, output_dir='output_crops', metadata_path=None, append=False):
    # Initialize JSON data structure
    if append and metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "type": "AI_EOTrainingDataset",
            "name": "Dataset for Beluga Whale Detection",
            "description": "Dataset for Beluga Whale Detection",
            "license": "",
            "version": "0.1.0",
            "createTime": datetime.now().strftime("%Y-%m-%d"),
            "providers": ["University of Chinese Academy of Sciences", "Fisheries and Oceans Canada"],
            "classes": ["adult", "junior", "calf"],
            "numberOfClasses": 3,
            "bands": ["panchromatic"],
            "amountOfTrainingTiles": 0,
            "amountOfCroppedData": 0,
            "amountOfTrainingLabels": 0,
            "tiles": []
        }

    for tile in tiles_with_targets:
        tile = tile[3:]
        pan_path = os.path.join(pan_dir, f'{tile}.jpg')
        if os.path.exists(pan_path):
            try:
                with rasterio.open(pan_path) as src:
                    print(f"Tile: {tile}  Size: {src.width} x {src.height}")
                    print(f"Bounds: {src.bounds}")

                    # Initialize tile metadata
                    tile_metadata = {
                        "tileURL": f"{tile}.jpg",
                        "dataSources": [dataSource],
                        "imagerySize": [src.width, src.height],
                        "cropSize": [crop_size, crop_size],
                        "numberOfCrops": 0,
                        "numberOfLabels": 0,
                        "windows": []
                    }

                    # Get whale positions in this tile
                    whale_positions = whale_gdf[whale_gdf['Photo-ID'] == f'CS_{tile}']
                    crop_info = {}

                    for _, whale in whale_positions.iterrows():
                        geom = whale.geometry
                        # print(geom)
                        if not geom.is_empty:
                            # Convert whale geometry to raster coordinates
                            x, y = geom.coords[0]
                            row, col = rowcol(src.transform, x, y)
                            # print(col, row)

                            # Determine crop window indices
                            crop_row = row // crop_size
                            crop_col = col // crop_size
                            window = Window(crop_col * crop_size, crop_row * crop_size, crop_size, crop_size)

                            # Read the crop window
                            crop = src.read([1, 2, 3], window=window)

                            # Save crop as a new jpeg file
                            crop_filename = f"{tile}_{crop_row}_{crop_col}.jpg"
                            crop_path = os.path.join(output_dir, crop_filename)
                            os.makedirs(os.path.dirname(crop_path), exist_ok=True)

                            with rasterio.open(crop_path, "w", driver="JPEG", height=crop.shape[1], width=crop.shape[2], count=3,
                                               dtype=crop.dtype, crs=src.crs, transform=src.window_transform(window)) as dest:
                                dest.write(crop[0], 1)  # Write the Red band
                                dest.write(crop[1], 2)  # Write the Green band
                                dest.write(crop[2], 3)  # Write the Blue band

                            # Calculate the point's position in the crop window
                            point_row = row - crop_row * crop_size
                            point_col = col - crop_col * crop_size
                            # print(type(point_row), type(point_col), type(row), type(col))

                            # Update crop information
                            if (crop_row, crop_col) not in crop_info:
                                crop_info[(crop_row, crop_col)] = {
                                    "cropID": [int(crop_row), int(crop_col)],
                                    "dataURL": crop_filename,
                                    "numberOfLabels": 0,
                                    "labels": []
                                }

                            crop_info[(crop_row, crop_col)]["labels"].append({
                                "class": whale["Species"],
                                "originID": whale["ID"],
                                "pointIndex": [int(point_row), int(point_col)]
                            })

                            crop_info[(crop_row, crop_col)]["numberOfLabels"] += 1

                    # Add crop info to tile metadata
                    for key, value in crop_info.items():
                        tile_metadata["windows"].append(value)

                    tile_metadata["numberOfCrops"] += len(crop_info)
                    tile_metadata["numberOfLabels"] += sum([info["numberOfLabels"] for info in crop_info.values()])

                    # Add tile metadata to JSON data
                    json_data["tiles"].append(tile_metadata)

                    json_data["amountOfTrainingTiles"] += 1
                    json_data["amountOfCroppedData"] += tile_metadata["numberOfCrops"]
                    json_data["amountOfTrainingLabels"] += tile_metadata["numberOfLabels"]
            except rasterio.errors.RasterioIOError as e:
                print(f"Error reading {pan_path}: {e}")

    # Save JSON data to a file
    json_output_path = os.path.join(output_dir, "metadata.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Saved metadata to {json_output_path}")