import rasterio
import pygrib
import geopandas as gpd
import pandas as pd
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time
from os import listdir
from os.path import isfile, join
from rasterio.transform import Affine

def extract_arrays(datestemp):
    # datestemp = temps.select(date = int(date))
    '''
    mylist = []
    for i in range(0, len(datestemp)):
        temp_array = datestemp[i].values
        mylist.append(temp_array)
    mylist
    '''
    data = np.array(datestemp)
    data_masked = ma.masked_where(data == 9999, data)
    min_temp = data_masked.min(axis=0)[0]
    max_temp = data_masked.max(axis=0)[0]
    mean_temp = data_masked.mean(axis=0)[0]
    print(min_temp.shape)
    return min_temp, max_temp, mean_temp


# MAKE GRID FROM TIFF ELEMENTS
def take_point_position(xa, ya):
    Xmin = 19.4100000
    Xmax = 28.3100000
    Ymin = 34.3900000
    Ymax = 41.8900000
    cols = 88
    rows = 74
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
    filepath = '/home/sg/Projects/FIrehub-model/tiffs/dew_temp/dewpoint_temp.tif'
    with rasterio.open(filepath) as src:
        print(src.crs)
        metadata = src.profile
    #metadata.update(
    #    transform=Affine(0.02, 0.0, 19.2,
    #                     0.0, 0.02, 34))
    with rasterio.open('/home/sg/Projects/FIrehub-model/tiffs/dew_temp/' + output + '.tif', 'w',
                       **metadata) as dst:
        dst.write(input.astype(rasterio.float64), 1)


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

def attribute_geodataframe_numpy(x, max_temp, min_temp, mean_temp,M,xcolumn,ycolumn):
    x_pos, y_pos = take_point_position(x[xcolumn], x[ycolumn])
    max_temp_values = -1000
    mean_temp_values = -1000
    min_temp_values = -1000
    if x_pos > 88 or y_pos > 74:
        return
    if max_temp[y_pos][x_pos]:
        max_temp_values = max_temp[y_pos][x_pos]
        mean_temp_values = mean_temp[y_pos][x_pos]
        min_temp_values = min_temp[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not max_temp[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= 88 - i and y_pos <= 74 - i:
                max_temp_values = np.mean(max_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                mean_temp_values = np.mean(mean_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                min_temp_values = np.mean(min_temp[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if max_temp_values != -1000:
                    break
    x[M] = max_temp_values
    x[M+1] = mean_temp_values
    x[M+2] = min_temp_values
    return (x)

def run_dew_temp(dataset_join,date,temps,q):
    start = time.time()
    print('Importing temperature data DONE in %s seconds'%(time.time() - start))
    #temps = pygrib.open("/home/sg/Projects/FIrehub-model/New_validation/dewpoint_temp.grib")
    datesarray = read_grib_data(temps, date)

    start = time.time()
    xcolumn = dataset_join.columns.get_loc('x')
    ycolumn = dataset_join.columns.get_loc('y')

    print('Importind dataset DONE in %s seconds'%(time.time() - start))
    #start = time.time()
    dataset_np = dataset_join.to_numpy()
    #print('Numping dataset DONE in %s seconds'%(time.time() - start))

    basket = np.empty((0, dataset_np.shape[1] + 3))

    N, M = dataset_np.shape
    dataset_np_temp = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 3))))

    #date = '20190813'
    datestemp = datesarray[date]

    min_temp = []
    max_temp = []
    mean_temp = []
    min_temp, max_temp, mean_temp = extract_arrays(datestemp)
    #temp_to_tif(min_temp,'minimum_'+date)
    #temp_to_tif(max_temp,'maximum_'+date)
    #temp_to_tif(mean_temp,'mean_'+date)

    start = time.time()
    my_list = []
    temperature_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_temp, max_temp, min_temp, mean_temp,M,xcolumn,ycolumn)
    print('Attribution DONE in %s seconds'%(time.time() - start))
    for array in temperature_arr:
        my_list.append(array.tolist())

    dataset_join['max_dew_temp'] = -1000
    dataset_join['mean_dew_temp'] = -1000
    dataset_join['min_dew_temp'] = -1000
    temp_df = gpd.GeoDataFrame(my_list)
    temp_df.columns = dataset_join.columns
    #temp_df = temp_df.dropna()
    temp_df['id'] = temp_df.id.astype('int64')
    #temp_df = temp_df[['id','max_dew_temp','mean_dew_temp','min_dew_temp']].copy()
    #temp_df.to_csv('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_temp.csv')
    q.put(temp_df)

'''
my_path = '/media/sg/91d62d44-8446-4f66-8327-bc09e774cbb1/home/df/Projects/Stella'
onlydirs = [f for f in listdir(my_path) if not isfile(join(my_path, f)) and '2019' in f]
temps = pygrib.open("/home/sg/Projects/FIrehub-model/New_validation/dewpoint_temp.grib")
for folder in onlydirs:
    files = [f for f in listdir(join(my_path,folder)) if 'wlst' in f]
    dates = [f[0:8] for f in files]
    datesarray = read_grib_data(temps, dates)
    for file in files:
        date = file[0:8]
        temp_df = run_dew_temp(datesarray,join(my_path,folder,file),date)
        temp_df.to_csv(join(my_path,folder,date+'_lst_dew.csv'))'''