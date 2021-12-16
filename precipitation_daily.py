import rasterio
import pygrib
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def take_point_position(xa, ya):
    Xmin = 18.9499999999999993
    Xmax = 28.0499999999999972
    Ymin = 33.9499999999999957
    Ymax = 42.0499999999999972
    cols = 91
    rows = 81
    # x, y = np.mgrid[Xmin:Xmax:complex(cols),Ymin:Ymax:complex(rows)]
    #    xa = 28.04999
    #    ya = 42.04999
    dx = xa - Xmin
    dy = Ymax - ya
    pixel = 0.1000000000000000056
    x_pos = int(dx / pixel)
    y_pos = int(dy / pixel)
    return (x_pos, y_pos)


def temp_to_tif(input, output):
    with rasterio.open('/home/sg/Projects/FIrehub-model/temps/' + fire_date[0:4] + '/' + output + fire_date + '.tif', 'w',
                       **metadata) as dst:
        dst.write(input, 1)


def read_grib_data(precs, dates):
    mylist = []
    dates_int = [int(i) for i in dates]
    precs.seek(0)
    for prec in precs:
        if prec.date in dates_int:
            mylist.append(prec.data()[0])
    data = np.array(mylist)
    print(data.shape)
    data_masked = ma.masked_where(data == 9999, data)
    datesar = np.sum(data_masked, axis=0)
    return datesar


def attribute_geodataframe_apply(ag_prec, row):
    x_pos, y_pos = take_point_position(row.geometry.x, row.geometry.y)
    if x_pos > 90 or y_pos > 80:
        return row

    for i in range(1, 5):
        if not ag_prec[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= 90 - i and y_pos <= 80 - i:
            precip_values = np.mean(ag_prec[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
            if precip_values:
                break
    else:
        precip_values = ag_prec[y_pos][x_pos]
    row['prcp'] = precip_values
    return row

start = time.time()
precs = pygrib.open("/home/sg/Downloads/downloadTotalPrecipitation2010_2019.grib")
dur = time.time() - start
print('Reading precipitation data DONE in %s seconds'%dur)
start = time.time()
dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset4_greece_20180723/centr_lu_with_DEM/centr_lu_with_Data.shp')
print('Reading dataset DONE in %s seconds'%(time.time() - start))
start = time.time()
dataset_join = dataset_join.to_crs('EPSG:4326')
print('Reprojecting dataset DONE in %s seconds'%(time.time() - start))

dates = []
fire_date = '20180723'
firedate = datetime.strptime(fire_date, "%Y%m%d")
start_date = firedate - timedelta(days=7)
end_date = firedate - timedelta(days=1)
for dt in daterange(start_date, end_date):
    dates.append(dt.strftime("%Y%m%d"))

start = time.time()
datesarr = read_grib_data(precs,dates)
print('Building dataset array Done in %s seconds'%(time.time() - start))

start = time.time()
a=dataset_join.apply(lambda row: attribute_geodataframe_apply(datesarr, row), axis = 1)
print('Attribution DONE in %s seconds'%(time.time() - start))

filepath = '/home/sg/Projects/FIrehub-model/New_validation/complementary_files/wind_grib_sample.tif'
with rasterio.open(filepath) as src:
    metadata = src.profile
temp_to_tif(datesarr,'precipitation')