from datetime import datetime
import requests
import json
import shapely
import geopandas as gpd
import pandas as pd
import zipfile
import os
from tqdm import tqdm
import glob
import rasterio
import rasterio.merge
from rasterio.mask import mask
import earthpy.plot as ep
import time
import numpy as np

def parseDateTime(title):
    return datetime.strptime(title.split("_")[2], "%Y%m%dT%H%M%S")

def makeCrs(title):
    return f"+proj=utm +zone={int(title.split('_')[5][1:3])} +datum=WGS84 +units=m +no_defs +type=crs"


class Sentinel_2:
    url = "https://scihub.copernicus.eu/dhus/search"

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.mosaic_red = None
        self.mosaic_nir = None

    def search_datasets(self, aoi, dateBounds, rows):

        self.aoi = aoi
        self.boundary = shapely.box(*aoi.bounds.values[0])
        self.dateBounds = dateBounds
        self.rows = rows
        tileInfo = {
            "tile_id": [],
            "acquisition_date": [],
            "tile_number": [],
            "link": [],
            "coords": [],
            "area": [],
            "cloudcoverpercentage": [],
            "crs" : []
        }
        
        self.params = {
            # search for Sentinel-2 Level-1C data
            "q": f'platformname:Sentinel-2 AND producttype:S2MSI1C AND footprint:"Intersects({str(self.boundary)})" AND beginPosition:[{self.dateBounds[0]} TO {self.dateBounds[1]}] AND endPosition:[{self.dateBounds[0]} TO {self.dateBounds[1]}]',
            "start": 0,  # start at the beginning of the result set
            "rows": self.rows,  # return up to 50 results
            "format": "json"
        }

        print(f"""
            filter bounds(BBox) :\t {aoi.bounds.values[0]}
            filter date         :\t {" TO ".join(self.dateBounds)}
            search results      :\t {self.rows}
            """)

        response = requests.get(Sentinel_2.url, params=self.params, auth=(self.username,self.password))

        print(response.status_code)
        if response.status_code == 200:
            # Parse the response as JSON
            data = response.json()

            for product in data["feed"].get('entry', []):
                title = product["title"]
                link = [item.get('href', '') for item in product['link'] if item.get('rel', '') == 'alternative'][0]
                footprint = [item.get('content', '') for item in product['str'] if item.get('name', '') == 'footprint'][0]
                cloudcover = product['double'].get('content')

                if not footprint:
                    continue

                poly = shapely.wkt.loads(footprint)

                if shapely.intersects(self.boundary, poly):
                    tileInfo['coords'].append(poly)
                    tileInfo["tile_id"].append(title)
                    tileInfo["link"].append(f"{link}$value")
                    tileInfo["acquisition_date"].append(parseDateTime(title))
                    tileInfo["tile_number"].append("_".join(title.split("_")[3:6]))
                    tileInfo["area"].append(poly.area)
                    tileInfo["cloudcoverpercentage"].append(float(cloudcover))
                    tileInfo['crs'].append(makeCrs(title))

                    print(f"{title}")

        df = pd.DataFrame(tileInfo)
        df_shapely = gpd.GeoDataFrame(df, geometry='coords')
        df_shapely.crs = 'EPSG:4326'

        if not df_shapely.empty:
            df_shapely.boundary.plot() 
        else:
            print("No tiles Found!")
        self.df_shapely = df_shapely
        return df_shapely
    
    #----------------------------------------------------------------------------

    def downloadTiles(self, shapely_df=gpd.GeoDataFrame(), stote_dir='./data_store'):
        
        # if df_shapely is not passed as param, download all tiles
        if not shapely_df.empty:
            self.df_shapely = shapely_df
            
        # Selection of CRS based on area occupancy of tiles
        self.crs = self.df_shapely.groupby("crs").sum('area').reset_index().sort_values('area',ascending=False)[:1].values[0][0]

        # reproject geodataframe to defined crs
        self.aoi = self.aoi.to_crs(self.crs)
        
        downloaded_files = []
        for i, items in self.df_shapely.iterrows():
            dictItems = dict(items)

            fileName = f"""{stote_dir}/{dictItems["tile_id"]}.zip"""
            print(dictItems["tile_id"])

            with requests.get(dictItems["link"], stream=True, auth=(self.username, self.password)) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                progress_bar = tqdm(
                    total=total_size, unit='iB', unit_scale=True)
                with open(fileName, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                progress_bar.close()

            downloaded_files.append(fileName)
        return downloaded_files
    
    #--------------------------------------------------------------------------------

    def extractZip(self,downloaded_files =[], output_dir="./data_store"):
        
        extracted_files = []
        for file in downloaded_files:
            with zipfile.ZipFile(file, 'r') as f:
                try:
                    f.extractall(output_dir)
                    extracted_files.append(file.replace(".zip", ".SAFE"))
                    os.remove(file) # after extraction zip delete the zip 
                    print(f"{file} - file unzipped")
                    
                except BadZipfile:
                    print(f"{file} - is not a good zip, is not unzipped")
                    
        return extracted_files
    
    #--------------------------------------------------------------------------------

    def mosaic(self, extracted_files = []):
        redSrcs = []
        nirSrcs = []
        for extracted_file in extracted_files:
            
            red_path = glob.glob(
                f"{extracted_file}/GRANULE/*/IMG_DATA/*_B04.jp2")[0]
            nir_path = glob.glob(
                f"{extracted_file}/GRANULE/*/IMG_DATA/*_B08.jp2")[0]

            redSrcs.append(rasterio.open(red_path))
            nirSrcs.append(rasterio.open(nir_path))
            
        print(redSrcs,nirSrcs)
        
        self.mosaic_red, self.out_transform_red = rasterio.merge.merge(
            redSrcs) if len(redSrcs) != 1 else (redSrcs[0], redSrcs[0].meta.transform)
        self.mosaic_nir, out_transform_nir = rasterio.merge.merge(nirSrcs) if len(
            nirSrcs) != 1 else (nirSrcs[0], nirSrcs[0].meta.transform)

        print("mosaic done")
        
    #--------------------------------------------------------------------------------

    def computeNDVI(self, output_dir="./data_store"):

        if self.mosaic_red.any() and self.mosaic_nir.any():

            red = self.mosaic_red[0].astype(float)
            nir = self.mosaic_nir[0].astype(float)

            NDVI = (nir-red)/(nir+red)
            red = None
            nir = None
            
            meta = {'driver': 'GTiff',
                    'width': self.mosaic_red.shape[2],
                    'height': self.mosaic_red.shape[1],
                    'count': 1,
                    'dtype': rasterio.float32,
                    'crs' : self.crs,
                    'transform': self.out_transform_red}
            
            filename = f'{output_dir}/{int(time.time())}_ndvi.tif'
            with rasterio.open(filename, 'w', **meta) as f:
                f.write(NDVI.astype(rasterio.float32), 1)
            
            return filename

        else:
            print('run mosaic function first')
            
    #-----------------------------------------------------------------------
    
    def clipByAoi(self,inputFilePath):
        
            raster = rasterio.open(inputFilePath)
            clipped_raster, clipped_transform = mask(
                raster, [self.aoi.geometry.values[0]], crop=True,nodata=np.NaN)

            clipped_meta = raster.meta.copy()
            clipped_meta.update({
                'height': clipped_raster.shape[1],
                'width': clipped_raster.shape[2],
                'transform': clipped_transform
            })

            with rasterio.open(inputFilePath.replace("ndvi",'clipped_ndvi'), 'w', **clipped_meta) as f:
                f.write(clipped_raster)


