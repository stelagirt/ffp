import json
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


def find_img(y, x):  # x=firedate, y=row
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
        image_name = list(filter(lambda a: a.split(".")[2] == y[9], hdf_dict[min_date]))
        return (image_name[0])
    except:
        print("Exception occured")

def nearest(data, min_date_str, hdf_dict,M):
    image_name = list(filter(lambda a: a.split(".")[2] == data[9], hdf_dict[min_date_str]))
    data[M] = image_name[0]
    return(data)


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
    return xmin, ymax, pixel_x, pixel_y, data_masked


def take_point_position(xmin, ymax, pixel_x, pixel_y, xa, ya):
    dx = float(xa) - xmin
    # print(dx)
    dy = ymax - float(ya)
    # print(dy)
    x_pos = int(dx / pixel_x)
    y_pos = int(dy / pixel_y)
    return x_pos, y_pos


def temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image):
    output = '/home/sg/Projects/FIrehub-model/Bartsotas/dataset_07072019/' + str(image) + '.tif'
    wgs84 = {'init': 'epsg:4326'}
    transform = from_origin(xmin, ymax, pixel_x, pixel_y)
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
                                height=ndvi.shape[0], width=ndvi.shape[1],
                                count=1, dtype=str(ndvi.dtype),
                                crs=wgs84,
                                transform=transform)
    new_dataset.write(ndvi, 1)
    new_dataset.close()


def attribute_geodataframe_numpy(x, xmin, ymax, pixel_x, pixel_y, ndvi,M):
    x_pos, y_pos = take_point_position(xmin, ymax, pixel_x, pixel_y, x[10].x, x[10].y)
    ndvi_values = -1000
    # print('Before%s'%x[28])
    if x_pos > ndvi.shape[1] - 1 or y_pos > ndvi.shape[0] - 1:
        return
        # print(x_pos,y_pos)
    # if x_pos > ndvi.shape[1]-1 or y_pos > ndvi.shape[0]-1:
    if ndvi[y_pos][x_pos]:
        ndvi_values = ndvi[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not ndvi[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= ndvi.shape[1] - i and y_pos <= \
                    ndvi.shape[0] - i:
                ndvi_values = np.mean(ndvi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if ndvi_values != -1000:
                    break
    x[M] = ndvi_values
    return (x)


def image_in_greece(tile, last_images):
    image = last_images[tile]
    return (image)


def run_ndvi(dataset_join,q):
    #dataset_join is the Greek points with the tile info for each point
    #dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')
    with open("/home/sg/test_modis_download/archive/most_recent.json") as f:
        last_images = json.load(f)

    dataset_join['image'] = dataset_join.apply(lambda x: image_in_greece(x['tile'], last_images), axis=1)

    dataset_np = dataset_join.to_numpy()
    images = last_images.values()
    N, M = dataset_np.shape
    dataset_np_ndvi = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 1))))

    my_path = "/home/sg/Projects/FIrehub-model/tiffs/ndvi/2019/ndvi_2019"
    os.chdir('/home/sg/test_modis_download/archive')
        # basket = np.array([])
        #basket = np.empty((0, dataset_np.shape[1] + 1))

    start_time = time.time()

    i = 0
    my_list = []
    for image in images:
        print(i+1, image)
        xmin, ymax, pixel_x, pixel_y, ndvi = import_hdf(image)
        #temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image)
        ndvi_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_ndvi[dataset_np_ndvi[:, M-1] == image],
                                           xmin, ymax,
                                           pixel_x, pixel_y, ndvi,M)
        for array in ndvi_arr:
            my_list.append(array.tolist())
            # basket = np.append(basket, [array], axis=0)
        i += 1


    dataset_join['ndvi'] = -1000
    ndvi_df = gpd.GeoDataFrame(my_list)#columns=dataset_join.columns)
    ndvi_df.columns = dataset_join.columns
    ndvi_df['id'] = ndvi_df.id.astype('float')
    #ndvi_df = ndvi_df.groupby('id').max().reset_index()
    #ndvi_df = ndvi_df.dropna()
    #ndvi_df.to_csv('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_ndvi.csv')
    #q.put(ndvi_df)
    #ndvi_df_nonmasked = ndvi_df[ndvi_df.ndvi != np.ma]
    # ndvi_df['firedate'] = ndvi_df['firedate'].astype(str)
    #ndvi_df_nonmasked.to_file('/home/sg/Projects/FIrehub-model/dataset4_greece_20180723/greece_with_ndvi')
    q.put(ndvi_df)

#dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')

#ndvi_df = run_ndvi(dataset_join)