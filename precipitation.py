import rasterio
import pygrib
import psycopg2
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import numpy.ma as ma
from datetime import datetime


def extract_arrays(datestemp):
    # datestemp = temps.select(date = int(date))
    mylist = []
    for i in range(0, len(datestemp)):
        temp_array = (datestemp[i][0])
        mylist.append(temp_array)
    mylist
    data = np.array(mylist)
    data_masked = ma.masked_where(data == 9999, data)
    min_temp = data_masked.min(axis=0)
    max_temp = data_masked.max(axis=0)
    mean_temp = data_masked.mean(axis=0)
    print(min_temp.shape)
    return min_temp, max_temp, mean_temp


# MAKE GRID FROM TIFF ELEMENTS
def take_point_position(xa, ya):
    Xmin = 18.9499999999999993
    Xmax = 28.0499999999999972
    Ymin = 33.9499999999999957
    Ymax = 42.0499999999999972
    cols = 91
    rows = 81
    x, y = np.mgrid[Xmin:Xmax:complex(cols), Ymin:Ymax:complex(rows)]
    #    xa = 28.04999
    #    ya = 42.04999
    dx = xa - Xmin
    dy = Ymax - ya
    pixel = 0.1000000000000000056
    x_pos = int(dx / pixel)
    y_pos = int(dy / pixel)
    return (x_pos, y_pos)


def attribute_geodataframe(date, max_temp, min_temp, mean_temp, centr_msg_date_wgs):
    for index, row in centr_msg_date_wgs[centr_msg_date_wgs.firedate_g == date].iterrows():
        x_pos, y_pos = take_point_position(row.geometry.x, row.geometry.y)
        if x_pos > 80 or y_pos > 90:
            continue
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'max_temp'] = \
        max_temp[y_pos][x_pos]
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'min_temp'] = \
        min_temp[y_pos][x_pos]
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'mean_temp'] = \
        mean_temp[y_pos][x_pos]
    return (centr_msg_date_wgs)


def temp_to_tif(input, output):
    temp_transform = metadata["transform"]
    temp_crs = metadata["crs"]
    # View spatial attributes
    # max_temp_transform, max_temp_crs
    # View the type of data stored
    # type(max_temp), max_temp.dtype
    with rasterio.open('/home/sg/Projects/FIrehub-model/temps/' + str(year) + '/' + output + date + '.tif', 'w',
                       **metadata) as dst:
        dst.write(input, 1)


def read_grib_data(temps, dates):
    temps.seek(0)
    checkdate = datetime.now()
    datesar = {}
    for tmp in temps:
        if checkdate != tmp.validDate:
            checkdate = tmp.validDate
            checkdate_st = checkdate.strftime('%Y%m%d')
            if checkdate_st in dates:
                datesar[checkdate_st] = []
        if tmp.validDate == checkdate and checkdate_st in dates:
            datesar[checkdate_st].append(tmp.data())
    return datesar

def attribute_geodataframe_numpy(x, max_temp, min_temp, mean_temp):
    x_pos, y_pos = take_point_position(x[3], x[4])
    max_temp_values = -1000
    mean_temp_values = -1000
    min_temp_values = -1000
    if x_pos > 90 or y_pos > 80:
        return
    if max_temp[y_pos][x_pos]:
        max_temp_values = max_temp[y_pos][x_pos]
        mean_temp_values = mean_temp[y_pos][x_pos]
        min_temp_values = min_temp[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not max_temp[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= 90 - i and y_pos <= 80 - i:
                max_temp_values = np.mean(max_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                mean_temp_values = np.mean(mean_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                min_temp_values = np.mean(min_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if max_temp_values != -1000:
                    break
    x[6] = max_temp_values
    x[7] = mean_temp_values
    x[8] = min_temp_values
    return (x)

temps = pygrib.open("/home/sg/Downloads/downloadPrecipitationALL.grib")
no_fire = gpd.read_file('/home/sg/Projects/FIrehub-model/no_fire_90/no_fire_120_final/no_fire_120_final.shp')
'''
no_fire_poly = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_alex/no fire 90/nofire90.shp')
no_fire = no_fire_poly.drop(columns = 'geometry')
greece_full = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset4_greece_20180723/centr_lu_with_Data/centr_lu_with_Data.shp')
greece_full = greece_full.to_crs(crs = {'init': 'epsg:4326'})
no_fire_static = no_fire.merge(greece_full,on = "id")
no_fire_static['x'] = no_fire_static.geometry.x
no_fire_static['y'] = no_fire_static.geometry.y
#For non fire points --2
no_fire_static['firedate'] = pd.to_datetime(no_fire_static['firedate'])
no_fire_static['firedate_g'] = no_fire_static['firedate'].dt.strftime('%Y%m%d')
'''

dataset_np = no_fire.to_numpy()
dates = []

dates_full = np.array(no_fire.firedate_g.drop_duplicates())

N, M = dataset_np.shape
dataset_np_rain = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 1))))


print('Start: %s'%datetime.now())
my_list = []
for year in range(2010,2019):
    dates = dates_full[[date.startswith(str(year)) for date in dates_full]]
    #    centr_msg_date_wgs_upd = centr_msg_date_wgs.copy()

    datesar = read_grib_data(temps,dates)

    for date in dates:
        rain = []
        try:
            rain = extract_arrays(datesar[date])
        #    temp_to_tif(min_temp,'minimum')
         #   temp_to_tif(max_temp,'maximum')
         #   temp_to_tif(mean_temp,'mean')
            no_fire_static = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np[dataset_np[:, 9] == date],
                                       xmin, ymax,
                                       pixel_x, pixel_y, ndvi)
            for array in no_fire_static:
                my_list.append(array)
            print("{} ok!".format(date))
        except:
            print("No product for %s"%date)
print('End: %s'%datetime.now())