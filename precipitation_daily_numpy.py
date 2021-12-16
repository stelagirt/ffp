import rasterio
import pygrib
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time


def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
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

def attribute_geodataframe_numpy(x, ndvi,M):
    x_pos, y_pos = take_point_position(x[11], x[12])
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

def run_prec(dataset_join,fire_date,precs,q):

    start = time.time()
    #precs = pygrib.open("/home/sg/Downloads/downloadTotalPrecipitation2010_2019.grib")
    dur = time.time() - start
    print('Importing precipitation data DONE in %s seconds'%dur)
    #start = time.time()
    #dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_id_geom.shp')
    #print(dataset_join.crs)
    #print('Reading dataset DONE in %s seconds'%(time.time() - start))
    #start = time.time()
    dataset_np = dataset_join.to_numpy()
    #print('Numping dataset DONE in %s seconds'%(time.time() - start))

    basket = np.empty((0, dataset_np.shape[1] + 1))

    N, M = dataset_np.shape
    dataset_np_prec = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 1))))

    dates = []
    #fire_date = '20190813'
    firedate = datetime.strptime(fire_date, "%Y%m%d")
    start_date = firedate - timedelta(days=7)
    end_date = firedate - timedelta(days=1)
    for dt in daterange(start_date, end_date):
        dates.append(dt.strftime("%Y%m%d"))

    start = time.time()
    datesarr = read_grib_data(precs,dates)
    print('Building dataset array Done in %s seconds'%(time.time() - start))

    start = time.time()
    my_list = []
    precipitation_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_prec, datesarr,M)
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