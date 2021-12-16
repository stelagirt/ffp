import rasterio
import pygrib
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time


def extract_arrays(datestemp):
    # datestemp = temps.select(date = int(date))
    mylist = []
    for i in range(0, len(datestemp)):
        temp_array = datestemp[i].values
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

def attribute_geodataframe_numpy(x, max_temp, min_temp, mean_temp,M):
    x_pos, y_pos = take_point_position(x[11], x[12])
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
    x[M] = max_temp_values
    x[M+1] = mean_temp_values
    x[M+2] = min_temp_values
    return (x)


start = time.time()
temps = pygrib.open("/home/sg/Downloads/downloadTemperatureALL.grib")
print('Importing temperature data DONE in %s seconds'%(time.time() - start))

start = time.time()
dataset_join = gpd.read_file('/home/sg/Projects/vravrona/zonal/grid_static_features_2011_wgs.shp')
dataset_join['x'] = dataset_join.geometry.x
dataset_join['y'] = dataset_join.geometry.y
print('Importind dataset DONE in %s seconds'%(time.time() - start))
#start = time.time()
dataset_np = dataset_join.to_numpy()
#print('Numping dataset DONE in %s seconds'%(time.time() - start))

basket = np.empty((0, dataset_np.shape[1] + 3))

N, M = dataset_np.shape
dataset_np_temp = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 3))))

date = '20110817'
datestemp = temps.select(date = int(date))

min_temp = []
max_temp = []
mean_temp = []
min_temp, max_temp, mean_temp = extract_arrays(datestemp)
#temp_to_tif(min_temp,'minimum')
#temp_to_tif(max_temp,'maximum')
#temp_to_tif(mean_temp,'mean')

start = time.time()
my_list = []
temperature_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_temp, max_temp, min_temp, mean_temp,M)
print('Attribution DONE in %s seconds'%(time.time() - start))
for array in temperature_arr:
    my_list.append(array.tolist())

dataset_join['max_temp'] = -1000
dataset_join['min_temp'] = -1000
dataset_join['mean_temp'] = -1000
temp_df = gpd.GeoDataFrame(my_list)
temp_df.columns = dataset_join.columns
#temp_df = temp_df.dropna()
#temp_df['id'] = temp_df.id.astype('int64')
#temp_df = temp_df[['id','max_temp','min_temp','mean_temp']].copy()
temp_df.to_file('/home/sg/Projects/vravrona/zonal/clima/17082011_temp')


