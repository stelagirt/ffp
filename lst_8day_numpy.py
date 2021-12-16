import pandas as pd
import datetime
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
from shapely import geometry

def nearest(data, min_date_str, hdf_dict,M):
    image_name = list(filter(lambda a: a.split(".")[2] == data[9], hdf_dict[min_date_str]))
    data[M] = image_name[0]
    return(data)


def import_hdf(images):
    day_im = [i for i in images if 'day' in i]
    night_im = [i for i in images if 'night' in i]
    ds_day = gdal.Open(day_im[0])
    ds_night = gdal.Open(night_im[0])
    gt = ds_day.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    pixel_x = abs(gt[1])
    pixel_y = abs(gt[5])
    lst_day = ds_day.ReadAsArray().astype(np.float32) * 0.02
    lst_night = ds_night.ReadAsArray().astype(np.float32) * 0.02
    data_day = ma.masked_where(lst_day == 0.0, lst_day)  # Added by Stella
    data_night = ma.masked_where(lst_night == 0.0, lst_night)
    return xmin, ymax, pixel_x, pixel_y, data_day, data_night

def take_point_position(xmin, ymax, pixel_x, pixel_y, xa, ya):
    dx = float(xa) - xmin
    # print(dx)
    dy = ymax - float(ya)
    # print(dy)
    x_pos = int(dx / pixel_x)
    y_pos = int(dy / pixel_y)
    return x_pos, y_pos


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


def attribute_geodataframe_numpy(x, xmin, ymax, pixel_x, pixel_y, M,ndvi,evi,xcolumn,ycolumn):
    x_pos, y_pos = take_point_position(xmin, ymax, pixel_x, pixel_y, x[xcolumn], x[ycolumn])
    ndvi_values = -1000
    evi_values = -1000
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
                if ndvi_values:
                    break
    if evi[y_pos][x_pos]:
        evi_values = evi[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not evi[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= evi.shape[1] - i and y_pos <= \
                    evi.shape[0] - i:
                evi_values = np.mean(evi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if evi_values:
                    break
    x[M] = ndvi_values
    x[M+1] = evi_values
    return (x)


def run_lst(dataset_join,date,q):
    #dataset_join is the Greek points with the tile info for each point
    #fields=['id','x','y']
    #dataset_join = pd.read_csv(file,usecols=fields)
    #dataset_join = gpd.read_file('/media/sg/91d62d44-8446-4f66-8327-bc09e774cbb1/home/df/Projects/lst_8day/nulls_20180814.shp')
    xcolumn = dataset_join.columns.get_loc('x')
    ycolumn = dataset_join.columns.get_loc('y')
    dataset_lst = dataset_join.to_numpy()
    N, M = dataset_lst.shape
    dataset_np_lst = np.hstack((dataset_lst, np.zeros((dataset_lst.shape[0], 2))))

    #fire_date = '20180801'
    fire_date_dt = dt.strptime(date,'%Y%m%d')
    my_path = "/users/pa21/sgirtsou/transfered_files/data/lst/composites_2020/"

    hdf_name = []
    for image in os.listdir(my_path):
        if image.endswith('tif') and str(fire_date_dt.year) in image:
            hdf_name.append(image)
    hdf_date = []
    for image in hdf_name:
        date = str(int(image.split("_")[0][1:8])+8)
        hdf_date.append(date)

    hdf = pd.DataFrame(list(zip(hdf_name, hdf_date)), columns =['files', 'dates'])
    hdf.dates = pd.to_datetime(hdf.dates, format = '%Y%j')
    hdf['dates1'] = hdf['dates'].dt.strftime('%Y%m%d')
    hdf_dict = {k: list(v) for k, v in hdf.groupby('dates1')['files']}

    items = [dt.strptime(k, '%Y%m%d') for k, v in hdf_dict.items() if (dt.strptime(k, '%Y%m%d') < fire_date_dt) and k.startswith(str(fire_date_dt.year))]
    min_date = min(items, key=lambda x: abs(x - fire_date_dt))
    min_date_str = dt.strftime(min_date, format = '%Y%m%d')

        #dataset_join['image'] = dataset_join.apply(lambda x: image_in_greece(x['tile']), axis=1)
        #dataset_join['image'] = dataset_join.apply(lambda x: find_img(fire_date_dt, x['tile']),axis=1)


    images = hdf_dict[min_date_str]

    #optimization with numpy
    os.chdir('/users/pa21/sgirtsou/transfered_files/data/lst/composites_2020/')
    start_time = time.time()

    xmin, ymax, pixel_x, pixel_y, lst_day, lst_night = import_hdf(images)
#    attribute_geodataframe_numpy(dataset_np_lst[0], xmin, ymax, pixel_x, pixel_y, M, lst_day, lst_night)
        # temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image)
    lst_day_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_lst,
                                       xmin, ymax,
                                       pixel_x, pixel_y,M, lst_day, lst_night,xcolumn,ycolumn)
    my_list=[]
    for array in lst_day_arr:
        my_list.append(array.tolist())
            # basket = np.append(basket, [array], axis=0)

    dataset_join['lst_day'] = -1000
    dataset_join['lst_night'] = -1000
    lst_df = gpd.GeoDataFrame(my_list)  # columns=dataset_join.columns)
    lst_df.columns = dataset_join.columns
    #lst_df = lst_df[lst_df.id != 'None']
    #lst_df['id'] = lst_df.id.astype('float')
    #ndvi_df = ndvi_df.groupby('id').max().reset_index()
    #lst_df = lst_df.dropna()
    #lst_df = lst_df[lst_df.lst_day!=-1000]
    #return lst_df
    #lst_df.to_csv(csv_file+'/'+fire_date+'_lst.csv')
    q.put(lst_df)
'''
csv_file = '/media/sg/91d62d44-8446-4f66-8327-bc09e774cbb1/home/df/Projects/Stella/outlier_2019'
for file in os.listdir(csv_file):
    if file.startswith('2'):
        print(file)
        os.chdir(csv_file)
        date = str(file)[0:8]
        run_lst(file, date)'''
