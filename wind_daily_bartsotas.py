import rasterio
from rasterio.transform import Affine
import pygrib
import geopandas as gpd
import math
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time
from os import chdir


# import gdal

# FINDS MAX VALUE OF VELOCITY -- FINAL
def max_velocity(ucomp, vcomp):
    ulist = []
    vlist = []
    # res
    for i in range(0, len(ucomp)):
        uval = ucomp[i]
        vval = vcomp[i]
        ulist.append(uval)
        vlist.append(vval)

    ucomp = np.ma.stack(ulist)
    vcomp = np.ma.stack(vlist)
    res = Pythagorean(ucomp, vcomp)
    res = ma.masked_where(res == -9999, res)
    res_max = np.amax(res, axis=0)
    res_max = ma.masked_where(res_max == -9999, res_max)
    pos_max = np.argmax(res, axis=0)
    return (res, res_max, pos_max)


def hourly_direction(u, v):
    return 180 + (180 / math.pi) * math.atan2(u, v)


def daily_max_direction(ucomp, vcomp, pos_max):
    daily_direction = np.ma.zeros([len(ucomp), 400, 480], fill_value=0)
    daily_direction = ma.masked_where(daily_direction == 0, daily_direction)
    wdf = np.vectorize(hourly_direction)
    daily_direction = wdf(ucomp, vcomp)
    daily_direction = ma.masked_where(daily_direction == 225., daily_direction)
    daily_dir_cat = wind_direction(daily_direction)
    m, n = pos_max.shape
    I, J = np.ogrid[:m, :n]
    azim_max = daily_direction[pos_max, I, J]
    dir_max = wind_direction(azim_max)
    return daily_dir_cat, dir_max


def wind_direction(azim):
    wind_dir = np.ma.copy(azim)
    wind_dir[((wind_dir > 337.5) | (wind_dir <= 22.5)) & (~wind_dir.mask)] = int(1)  # N
    wind_dir[(wind_dir > 22.5) & (wind_dir <= 67.5) & (~wind_dir.mask)] = int(2)  # NE
    wind_dir[(wind_dir > 67.5) & (wind_dir <= 112.5) & (~wind_dir.mask)] = int(3)  # E
    wind_dir[(wind_dir > 112.5) & (wind_dir <= 157.5) & (~wind_dir.mask)] = int(4)  # SE
    wind_dir[(wind_dir > 157.5) & (wind_dir <= 202.5) & (~wind_dir.mask)] = int(5)  # S
    wind_dir[(wind_dir > 202.5) & (wind_dir <= 247.5) & (~wind_dir.mask)] = int(6)  # SW
    wind_dir[(wind_dir > 247.5) & (wind_dir <= 292.5) & (~wind_dir.mask)] = int(7)  # W
    wind_dir[(wind_dir > 292.5) & (wind_dir <= 337.5) & (~wind_dir.mask)] = int(8)  # NW
    return wind_dir


def compute_dom_values(daily_dir_cat, res):
    import operator
    dom_vel = np.zeros([res.shape[1], res.shape[2]])
    dom_dir = np.zeros([res.shape[1], res.shape[2]])
    for j in range(res.shape[1]):
        for k in range(res.shape[2]):
            value = None
            if daily_dir_cat[0, j, k]:
                b = daily_dir_cat[0:res.shape[0], j, k]
                a = np.bincount(list(b))
                value, freq = max(enumerate(a), key=operator.itemgetter(1))
                dom_dir[j, k] = value
            if value:
                c = res[0:res.shape[0],j,k][daily_dir_cat[0:res.shape[0], j, k] == value]
                dom_vel[j, k] = max(c)

    dom_dir = ma.masked_where(dom_dir == 0, dom_dir)
    dom_vel = ma.masked_where(dom_vel == 0, dom_vel)
    return dom_dir, dom_vel

def Pythagorean(a, b):
    c = np.sqrt(a ** 2 + b ** 2)
    return (c)

def extract_arrays(ucomp, vcomp):
    mylist_u = []
    mylist_v = []
    for i in range(0, len(ucomp)):
        temp_array_u = ucomp[i].values
        mylist_u.append(temp_array_u)
        temp_array_v = vcomp[i].values
        mylist_v.append(temp_array_v)
    u = np.array(mylist_u)
    u_masked = ma.masked_where(u == 9999, u)
    v = np.array(mylist_v)
    v_masked = ma.masked_where(v == 9999, v)
    res, res_max, pos_max = max_velocity(u_masked, v_masked)
    daily_dir_cat, dir_max = daily_max_direction(u_masked, v_masked, pos_max)
    dom_dir, dom_vel = compute_dom_values(daily_dir_cat, res)
    return dom_dir, dom_vel, res_max, dir_max

# MAKE GRID FROM TIFF ELEMENTS
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
        transform = Affine(0.02, 0.0, 19.2,
                                0.0, 0.02, 34))
    with rasterio.open('/home/sg/Projects/FIrehub-model/Bartsotas/dataset_07072019/'+output+'.tif', 'w',
                       **metadata) as dst:
        dst.write(input.astype(rasterio.float64), 1)


def read_grib_data(wind, dates):
    wind.seek(0)
    checkdate = datetime.now()
    datesar_u = {}
    datesar_v = {}
    dtype = 'u'
    for w in wind:
        if checkdate != w.validDate:
            checkdate = w.validDate
            checkdate_st = checkdate.strftime('%Y%m%d')
            if checkdate_st in dates:
                datesar_u[checkdate_st] = []
                datesar_v[checkdate_st] = []
        if w.validDate == checkdate and checkdate_st in dates and dtype == 'u':
            datesar_u[checkdate_st].append(w.data())
        elif w.validDate == checkdate and checkdate_st in dates and dtype == 'v':
            datesar_v[checkdate_st].append(w.data())
        dtype = 'v' if dtype == 'u' else 'u'
    return datesar_u, datesar_v


def attribute_geodataframe_numpy(x, res_max, dir_max, dom_vel, dom_dir,M):
    x_pos, y_pos = take_point_position(x[10].x, x[10].y)
    res_max_values = -1000
    dir_max_values = -1000
    dom_vel_values = -1000
    dom_dir_values = -1000
    if x_pos > res_max.shape[1] - 1 or y_pos > res_max.shape[0] - 1:
        return
    if res_max[y_pos][x_pos]:
        res_max_values = res_max[y_pos][x_pos]
        dir_max_values = dir_max[y_pos][x_pos]
        dom_vel_values = dom_vel[y_pos][x_pos]
        dom_dir_values = dom_dir[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not res_max[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= res_max.shape[1] - i and y_pos <= res_max.shape[0] - i:
                res_max_values = np.mean(res_max[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                dir_max_values = np.mean(dir_max[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                dom_vel_values = np.mean(dom_vel[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                dom_dir_values = np.mean(dom_dir[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if res_max_values != -1000:
                    break
    x[M] = res_max_values
    x[M+1] = dir_max_values
    x[M+2] = dom_vel_values
    x[M+3] = dom_dir_values
    return (x)

def run_wind(dataset_join,meteo_data,q):

    dataset_np = dataset_join.to_numpy()
    start = time.time()
    #dateswind = wind.select(name = ['10 metre U wind component', '10 metre V wind component'], date=int(date))
    #ucomp, vcomp = read_grib_data(wind, date)
    forecast = list(range(37, 61, 1))
    ucomp = meteo_data.select(name = '10 metre U wind component', endStep=forecast) #Τime 36 to 60
    vcomp = meteo_data.select(name='10 metre V wind component', endStep=forecast)

    dom_dir, dom_vel, res_max, dir_max = extract_arrays(ucomp, vcomp)

    temp_to_tif(dom_dir, 'dom_dir')
    temp_to_tif(dom_vel, 'dom_vel')
    temp_to_tif(res_max, 'res_max')
    temp_to_tif(dir_max, 'dir_max')

    print('Arrays built in %s seconds' % (time.time() - start))

    N, M = dataset_np.shape
    dataset_np_wind = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 4))))

    start = time.time()
    my_list = []
    wind_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_wind, res_max, dir_max, dom_vel, dom_dir,M)
    print('Attribution DONE in %s seconds' % (time.time() - start))
    for array in wind_arr:
        my_list.append(array.tolist())

    dataset_join['res_max'] = -1000
    dataset_join['dir_max'] = -1000
    dataset_join['dom_vel'] = -1000
    dataset_join['dom_dir'] = -1000
    wind_df = gpd.GeoDataFrame(my_list)
    wind_df.columns = dataset_join.columns
    wind_df = wind_df.dropna()
    wind_df['id'] = wind_df.id.astype('int64')
    wind_df = wind_df[['id','res_max','dir_max','dom_vel','dom_dir']].copy()
    #wind_df.to_csv('/home/sg/Projects/FIrehub-model/Bartsotas/greece_wind.csv')
    q.put(wind_df)

'''
if __name__ == "__main__":
    dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')
    #dataset_join['x'] = dataset_join.geometry.x
    #dataset_join['y'] = dataset_join.geometry.y
    date = (date.today() - timedelta(days=2))
    date_str = date.strftime("%Y%m%d")
    meteo = 'WRF-' + date_str + '.grb2'
    chdir('/home/sg/test_modis_download')
    meteo_data = pygrib.open(meteo)
    run_wind(dataset_join, meteo_data)
'''