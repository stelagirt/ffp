import pandas as pd
from datetime import datetime as dt
import os
import osr
from osgeo import gdal, ogr, gdalconst
import numpy as np
import numpy.ma as ma
import geopandas as gpd
import gdal
import rasterio
from rasterio.transform import from_origin
import time


def find_img(x, y):  # x=firedate, y=tile
    # print(x.year, x)
    new_dict = [k for k, v in hdf_dict.items() if (dt.strptime(k, '%Y%m%d') <= x) and k.startswith(str(x.year))]
    # print(new_dict)
    try:
        min_date = new_dict[0]
        min_days = x - dt.strptime(new_dict[0], '%Y%m%d')
        for item in new_dict:
            if x - dt.strptime(item, '%Y%m%d') < min_days:
                min_days = x - dt.strptime(item, '%Y%m%d')
                min_date = item
        image_name = list(filter(lambda a: a.split(".")[2] == y, hdf_dict[min_date]))
        return (image_name[0])
    except:
        print("Exception occured")


def import_hdf(hdf):
    ds = gdal.Open(hdf)
    t_srs = osr.SpatialReference()
    t_srs.ImportFromEPSG(4326)
    src_ds = gdal.Open(ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
    dst_wkt = t_srs.ExportToWkt()
    dswrap = gdal.Warp('', src_ds, dstSRS='EPSG:4326', outputType=gdal.GDT_Int16, format='VRT')
    xmin = dswrap.GetGeoTransform()[0]
    ymax = dswrap.GetGeoTransform()[3]
    pixel_x = abs(dswrap.GetGeoTransform()[1])
    pixel_y = abs(dswrap.GetGeoTransform()[5])
    hdf_ar = dswrap.ReadAsArray().astype(np.float)
    data_np = np.array(hdf_ar)
    data_m = ma.masked_where(data_np == -3000, data_np)  # Added by Stella
    data_masked = data_m * 0.0001  # Added by Stella
    return (xmin, ymax, pixel_x, pixel_y, data_masked)


def take_point_position(xmin, ymax, pixel_x, pixel_y, xa, ya):
    dx = float(xa) - xmin
    # print(dx)
    dy = ymax - float(ya)
    # print(dy)
    x_pos = int(dx / pixel_x)
    y_pos = int(dy / pixel_y)
    return (x_pos, y_pos)


def temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image):
    output = '/home/sg/Projects/FIrehub-model/New_validation/ndvi_tiffs/' + str(image) + '.tif'
    wgs84 = {'init': 'epsg:4326'}
    transform = from_origin(xmin, ymax, pixel_x, pixel_y)
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
                                height=ndvi.shape[0], width=ndvi.shape[1],
                                count=1, dtype=str(ndvi.dtype),
                                crs=wgs84,
                                transform=transform)
    new_dataset.write(ndvi, 1)
    new_dataset.close()


def attribute_geodataframe_numpy(x, xmin, ymax, pixel_x, pixel_y, ndvi,evi):
    x_pos, y_pos = take_point_position(xmin, ymax, pixel_x, pixel_y, x[22], x[23])
    ndvi_values = -1000
    # print('Before%s'%x[28])
    if x_pos > ndvi.shape[1] - 1 or y_pos > ndvi.shape[0] - 1:
        return
        # print(x_pos,y_pos)
    # if x_pos > ndvi.shape[1]-1 or y_pos > ndvi.shape[0]-1:
    if ndvi[y_pos][x_pos]:
        ndvi_values = ndvi[y_pos][x_pos]
        evi_values = evi[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not ndvi[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= ndvi.shape[1] - i and y_pos <= \
                    ndvi.shape[0] - i:
                ndvi_values = np.mean(ndvi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                evi_values = np.mean(evi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if ndvi_values != -1000:
                    break
    x[28] = ndvi_values
    x[29] = evi_values
    return (x)


fire_data = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_2nd/fire_points_ftpwm.shp')
no_fire_data = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_2nd/no_fire_points_ftpwm.shp')
dataset = pd.concat([fire_data, no_fire_data]).reset_index()
dates = pd.to_datetime(dataset['firedate']).dt.strftime('%Y%m%d')

my_path = "/home/sg/Projects/FIrehub-model/New_validation/ndvi"
hdf_name = []
for image in os.listdir(my_path):
    if image.endswith('hdf'):
        hdf_name.append(image)
hdf_date = []
for image in hdf_name:
    date = image.split(".")[1][1:8]
    hdf_date.append(date)

hdf = pd.DataFrame(list(zip(hdf_name, hdf_date)), columns=['files', 'dates'])
hdf.dates = pd.to_datetime(hdf.dates, format='%Y%j')
hdf['dates1'] = hdf['dates'].dt.strftime('%Y%m%d')

dataset['x'] = dataset.geometry.x
dataset['y'] = dataset.geometry.y

my_path = "/home/sg/Projects/FIrehub-model/New_validation/ndvi"

tiles = gpd.read_file('/home/sg/Projects/FIrehub-model/New_validation/ndvi_tiffs/tiles/tiles.shp')
dataset_join = gpd.sjoin(dataset, tiles, op='within')

dataset_join['firedate'] = pd.to_datetime(dataset_join['firedate'])
dataset_join['firedate_grib'] = dataset_join['firedate'].dt.strftime('%Y%m%d')
hdf_dict = {k: list(v) for k, v in hdf.groupby('dates1')['files']}

min_date = []
dataset_join['image'] = dataset_join.apply(lambda x: find_img(x['firedate'], x['tile']), axis=1)
images = list(filter(None, list(dataset_join.image.unique())))

# optimization with numpy

dataset_np = dataset_join.to_numpy()

os.chdir('/home/sg/Projects/FIrehub-model/New_validation/ndvi')
basket = np.array([])

start_time = time.time()

test_im = ['MOD13A1.A2015193.h20v05.006.2015304005355.hdf']

N, M = dataset_np.shape
dataset_np_ndvi = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 1))))
i = 0
my_list = []
for image in test_im:
    print(i, image)
    xmin, ymax, pixel_x, pixel_y, ndvi, evi = import_hdf(image)
    # temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image)
    ndvi_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_ndvi[dataset_np_ndvi[:, 27] == image],
                                   xmin, ymax,
                                   pixel_x, pixel_y, ndvi)
    for array in ndvi_arr:
        my_list.append(array.tolist())
        # basket = np.append(basket, [array], axis=0)
    i += 1

    # basket = np.append(basket,temp)
ndvi_df = gpd.GeoDataFrame(my_list)
dataset_join['ndvi'] = -1000
ndvi_df.columns = dataset_join.columns
ndvi_df['firedate'] = ndvi_df['firedate'].astype(str)
ndvi_df.to_file('//home/sg/Projects/FIrehub-model/New_validation/ndvi_results/ndvi_df1')

elapsed_time = time.time() - start_time
print(elapsed_time)

print(xmin, ymax)
