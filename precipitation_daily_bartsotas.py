import rasterio
import pygrib
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time
from os import chdir
from rasterio.transform import Affine

def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def take_point_position(xa, ya):
    Xmin = 19.2
    Xmax = 28.49
    Ymin = 34
    Ymax = 41.99
    cols = 480
    rows = 400
    dx = xa - Xmin
    dy = ya - Ymin
    pixel = 0.02
    x_pos = int(dx / pixel)
    y_pos = int(dy / pixel)
    return (x_pos, y_pos)


def temp_to_tif(input, output):
    filepath = '/home/sg/Projects/FIrehub-model/Bartsotas/example.tif'
    with rasterio.open(filepath) as src:
        print(src.crs)
        metadata = src.profile
    metadata.update(
        transform=Affine(0.02, 0.0, 19.2,
                         0.0, 0.02, 34))
    with rasterio.open('/home/sg/Projects/FIrehub-model/Bartsotas/dataset_07072019/' + output + '.tif', 'w',
                       **metadata) as dst:
        dst.write(input.astype(rasterio.float64), 1)


def read_grib_data():
    chdir('/home/sg/test_modis_download')
    my_list = []
    for i in range(1, 8):
        d_date = (date.today() - timedelta(days=i)).strftime("%Y%m%d")
        meteo = 'WRF-' + d_date + '.grb2'
        minus_data = pygrib.open(meteo).select(name='Total Precipitation', endStep=12)
        meteo_data = pygrib.open(meteo).select(name='Total Precipitation', endStep=36)
        prec = meteo_data[0].data()[0] - minus_data[0].data()[0]
        my_list.append(prec)
    data = np.array(my_list)
    print(data.shape)
    data_masked = ma.masked_where(data == 9999, data)
    final_array = np.sum(data_masked, axis=0)
    return final_array

def attribute_geodataframe_numpy(x, ndvi,M):
    x_pos, y_pos = take_point_position(x[10].x, x[10].y)
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

def run_prec(dataset_join,q):
    dataset_np = dataset_join.to_numpy()

    N, M = dataset_np.shape
    dataset_np_prec = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 1))))

    start = time.time()
    final_arr = read_grib_data()
    temp_to_tif(final_arr, 'rain')
    print('Building dataset array Done in %s seconds'%(time.time() - start))

    start = time.time()
    my_list = []
    precipitation_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_prec, final_arr,M)
    for array in precipitation_arr:
        my_list.append(array.tolist())

    print('Attribution DONE in %s seconds'%(time.time() - start))

    dataset_join['prcp'] = -1000
    prcp_df = gpd.GeoDataFrame(my_list)
    prcp_df.columns = dataset_join.columns
    prcp_df = prcp_df.dropna()
    prcp_df['id'] = prcp_df.id.astype('int64')
    prcp_df = prcp_df[['id','prcp']]
    #prcp_df.to_csv('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_rain.csv')
    q.put(prcp_df)

    #filepath = '/home/sg/Projects/FIrehub-model/New_validation/complementary_files/wind_grib_sample.tif'
    #with rasterio.open(filepath) as src:
    #    metadata = src.profile
    #temp_to_tif(datesarr,'precipitation')
