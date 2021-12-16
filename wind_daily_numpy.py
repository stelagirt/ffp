import rasterio
import pygrib
import geopandas as gpd
import math
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time


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
    daily_direction = np.ma.zeros([len(ucomp), 81, 91], fill_value=0)
    daily_direction = ma.masked_where(daily_direction == 0, daily_direction)
    wdf = np.vectorize(hourly_direction)
    daily_direction = wdf(ucomp, vcomp)
    daily_dir_cat = wind_direction(daily_direction)
    #     for i in range(len(ucomp)):
    #             for j in range(81):
    #                 for k in range(91):
    #                     if ucomp[i,j,k]:
    #                         #print(ucomp[1].values[i,j])
    #                         a = (180/math.pi)*float(metpy.calc.wind_direction(ucomp[i,j,k]* units.meter / units.second, vcomp[i,j,k]* units.meter / units.second))
    #                         daily_direction[i,j,k] =float(a)

    m, n = pos_max.shape
    I, J = np.ogrid[:m, :n]
    azim_max = daily_direction[pos_max, I, J]
    dir_max = wind_direction(azim_max)
    return daily_dir_cat, dir_max


def wind_direction(azim):
    wind_dir = np.ma.copy(azim)
    # wind_dir
    # wind_dir.shape, azim[1].size
    # wind_dir

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
    b = []
    c = []
    dom_vel = np.zeros([81, 91])
    dom_dir = np.zeros([81, 91])
    # freq = np.zeros([81,91], dtype=np.complex_)
    m = 0
    # dom_dir = ma.masked_where(dom_dir == 0, dom_dir)
    for j in range(81):
        for k in range(91):
            for i in range(24):
                if daily_dir_cat[i, j, k]:
                    sub = daily_dir_cat[i, j, k]
                    b.append(sub)
                    m += 1
                else:
                    break
                #
                #print(b)
                a = np.bincount(b)
                value, freq = max(enumerate(a), key=operator.itemgetter(1))
                # print(value, freq)  # prints the dominant direction and its frequency
                b = []
                dom_dir[j, k] = value
                if daily_dir_cat[i, j, k] == value:
                    c.append(res[i, j, k])
                dom_vel[j, k] = max(c)

    dom_dir = ma.masked_where(dom_dir == 0, dom_dir)
    dom_vel = ma.masked_where(dom_vel == 0, dom_vel)
    return dom_dir, dom_vel


#        print(i,wind_dir[i,j,k])
#             a = np.bincount(wind_dir)
#             print(a)
#             b.append(a)

def Pythagorean(a, b):
    c = np.sqrt(a ** 2 + b ** 2)
    return (c)


def extract_arrays(ucomp, vcomp):
    # datestemp = temps.select(date = int(date))
    mylist_u = []
    mylist_v = []
    for i in range(0, len(ucomp)):
        temp_array_u = (ucomp[i][0])
        mylist_u.append(temp_array_u)
        temp_array_v = (vcomp[i][0])
        mylist_v.append(temp_array_v)
    u = np.array(mylist_u)
    u_masked = ma.masked_where(u == 9999, u)
    v = np.array(mylist_v)
    v_masked = ma.masked_where(v == 9999, v)
    res, res_max, pos_max = max_velocity(u_masked, v_masked)
    daily_dir_cat, dir_max = daily_max_direction(u_masked, v_masked, pos_max)
    dom_dir, dom_vel = compute_dom_values(daily_dir_cat, res)
    return (dom_dir, dom_vel, res_max, dir_max)

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


def attribute_geodataframe(date, res_max, dir_max, dom_vel, dom_dir, centr_msg_date_wgs):
    for index, row in centr_msg_date_wgs[centr_msg_date_wgs.firedate_g == date].iterrows():
        x_pos, y_pos = take_point_position(row.geometry.x, row.geometry.y)
        if x_pos > 80 or y_pos > 90:
            continue
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'res_max'] = res_max[y_pos][
            x_pos]
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'dir_max'] = dir_max[y_pos][
            x_pos]
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'dom_vel'] = dom_vel[y_pos][
            x_pos]
        centr_msg_date_wgs.loc[
            (centr_msg_date_wgs.index == index) & (centr_msg_date_wgs.firedate_g == date), 'dom_dir'] = dom_dir[y_pos][
            x_pos]
    return (centr_msg_date_wgs)


def temp_to_tif(input, output):
    temp_transform = metadata["transform"]
    temp_crs = metadata["crs"]
    # View spatial attributes
    # max_temp_transform, max_temp_crs
    # View the type of data stored
    # type(max_temp), max_temp.dtype
    with rasterio.open('/home/sg/Projects/FIrehub-model/wind/' + str(year) + '/' + output + date + '.tif', 'w',
                       **metadata) as dst:
        dst.write(input, 1)


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
    x_pos, y_pos = take_point_position(x[11], x[12])
    res_max_values = -1000
    dir_max_values = -1000
    dom_vel_values = -1000
    dom_dir_values = -1000
    if x_pos > 90 or y_pos > 80:
        return
    if res_max[y_pos][x_pos]:
        res_max_values = res_max[y_pos][x_pos]
        dir_max_values = dir_max[y_pos][x_pos]
        dom_vel_values = dom_vel[y_pos][x_pos]
        dom_dir_values = dom_dir[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not res_max[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= 90 - i and y_pos <= 80 - i:
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

def run_wind(dataset_join, date,wind,q):
    #date = '20190813'
    start = time.time()
    #wind = pygrib.open("/home/sg/Downloads/era5-land-wind-2010-2020.grb")
    print('Importing wind data DONE in %s seconds' % (time.time() - start))

    #start = time.time()
    #dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_id_geom.shp')
    #print('Importind dataset DONE in %s seconds' % (time.time() - start))
    #start = time.time()
    dataset_np = dataset_join.to_numpy()
    #print('Numping dataset DONE in %s seconds' % (time.time() - start))

    start = time.time()
    dateswind = wind.select(date=int(date))
    ucomp, vcomp = read_grib_data(wind, date)
    dom_dir, dom_vel, res_max, dir_max = extract_arrays(ucomp[date], vcomp[date])
    print('Arrays built in %s seconds' % (time.time() - start))

    basket = np.empty((0, dataset_np.shape[1] + 3))

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
    #wind_df.to_csv('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_wind.csv')
    q.put(wind_df)
