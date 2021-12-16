import time
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from datetime import datetime
import pygrib
import math
import metpy
import csv
from metpy.units import units

def extract_arrays(ucomp,vcomp):
    #datestemp = temps.select(date = int(date))
    mylist_u = []
    mylist_v = []
    for i in range (0,len(ucomp)):
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
    return dom_dir, dom_vel, res_max, dir_max

def read_grib_data(wind,dates):
    wind.seek(0)
    checkdate = datetime.now()
    datesar_u={}
    datesar_v={}
    dtype='u'
    for w in wind:
        if checkdate != w.validDate:
            checkdate = w.validDate
            checkdate_st = checkdate.strftime('%Y%m%d')
            if checkdate_st in dates:
                datesar_u[checkdate_st]=[]
                datesar_v[checkdate_st]=[]
        if w.validDate == checkdate and checkdate_st in dates and dtype=='u':
            datesar_u[checkdate_st].append(w.data())
        elif w.validDate == checkdate and checkdate_st in dates and dtype=='v':
            datesar_v[checkdate_st].append(w.data())
        dtype='v' if dtype=='u' else 'u'
    return datesar_u, datesar_v


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

def wind_direction2(u, v):
    return 180+(180/math.pi)*math.atan2(u,v)

def wind_direction3(u, v):
    return (180/math.pi)*metpy.calc.wind_direction(u * units.meter / units.second,
                             v * units.meter / units.second)

def daily_max_direction(ucomp, vcomp, pos_max):
    daily_direction = np.ma.zeros([len(ucomp), 81, 91], fill_value=0)
    daily_direction = ma.masked_where(daily_direction == 0, daily_direction)


    wdf = np.vectorize(wind_direction2)
#    wdf2 = np.vectorize(wind_direction3)

    #uvec = np.reshape(ucomp, 24*81*91)
    #vvec = np.reshape(vcomp, 24*81*91)
    daily_direction = wdf(ucomp, vcomp)
    daily_dir_cat = wind_direction(daily_direction)
    '''
    for i in range(len(ucomp)):
        for j in range(81):
            for k in range(91):
                if ucomp[i, j, k]:
                    # print(ucomp[1].values[i,j])
                    a = (180 / math.pi) * float(metpy.calc.wind_direction(ucomp[i, j, k] * units.meter / units.second,
                                                                          vcomp[i, j, k] * units.meter / units.second))
                    daily_direction[i, j, k] = float(a)
    '''
    m, n = pos_max.shape
    I, J = np.ogrid[:m, :n]
    azim_max = daily_direction[pos_max, I, J]
    dir_max = wind_direction(azim_max)
    return (daily_dir_cat, dir_max)


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
    dom_vel = np.zeros([81, 91])
    dom_dir = np.zeros([81, 91])
    # freq = np.zeros([81,91], dtype=np.complex_)
    m = 0
    # dom_dir = ma.masked_where(dom_dir == 0, dom_dir)
    for j in range(81):
        for k in range(91):
            b1 = daily_dir_cat[:, j, k].astype(int)
            #print(b1)
            a = np.bincount(b1)
            #print(a)
            value, freq = max(enumerate(a), key=operator.itemgetter(1))
            #print(value, freq)  # prints the dominant direction and its frequency
            c1 = res[daily_dir_cat[:,j, k]==value, j, k]
            if len(c1)>0:
                dom_dir[j,k] = float(value)
                dom_vel[j, k] = float(max(c1))

    dom_dir = ma.masked_where(dom_dir == 0, dom_dir)
    dom_vel = ma.masked_where(dom_vel == 0, dom_vel)
    #print(dom_dir)
    return (dom_dir, dom_vel)


#        print(i,wind_dir[i,j,k])
#             a = np.bincount(wind_dir)
#             print(a)
#             b.append(a)

def Pythagorean(a, b):
    c = np.sqrt(a ** 2 + b ** 2)
    return c

#MAKE GRID FROM TIFF ELEMENTS
def take_point_position(xa,ya):
    Xmin = 18.9499999999999993
    Xmax = 28.0499999999999972
    Ymin = 33.9499999999999957
    Ymax = 42.0499999999999972
    cols = 91
    rows = 81
    x, y = np.mgrid[Xmin:Xmax:complex(cols),Ymin:Ymax:complex(rows)]
#    xa = 28.04999
#    ya = 42.04999
    dx = xa - Xmin
    dy = Ymax - ya
    pixel = 0.1000000000000000056
    x_pos = int(dx/pixel)
    y_pos = int(dy/pixel)
    return(x_pos,y_pos)

def attribute_geodataframe_numpy(x, res_max, dir_max, dom_vel, dom_dir):
    x_pos, y_pos = take_point_position(x[7], x[8])
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

abs_time = time.time()
centr_msg_date_wgs = gpd.read_file('/home/sg/Projects/FIrehub-model/no_fire_90/no_fire_static_temperature/no_fire_static_temperature.shp')
dataset_join = centr_msg_date_wgs[['id', 'firedate','firedate_g', 'max_temp', 'min_temp','mean_temp', 'geometry']].copy()
dataset_join['x'] = dataset_join.geometry.x
dataset_join['y'] = dataset_join.geometry.y
wind = pygrib.open("/home/sg/Downloads/era5-land-wind-2010-2020.grb")
#wind = pygrib.open("/home/sg/Downloads/adaptor.mars.internal-1581436007.5929372-29131-1-c01c3f4d-c21f-4454-b89c-9dd2e92dc733.grib")
#centr_msg_date_wgs = gpd.read_file('/home/sg/Projects/FIrehub-model/Dataset_1st/temperatures/centr_msg_date_wgs_temps_final25.shp')
dates_full = np.array(centr_msg_date_wgs.firedate_g.drop_duplicates())
print('Dataset loaded in %s minutes' % ((time.time() - abs_time)/60))

start = time.time()
dataset_np = dataset_join.to_numpy()
print('Numping dataset DONE in %s seconds' % (time.time() - start))

N, M = dataset_np.shape
dataset_np_wind = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 4))))
my_list = []
for year in range(2010, 2019):
    dates = dates_full[[date.startswith(str(year)) for date in dates_full]]
    #    centr_msg_date_wgs_upd = centr_msg_date_wgs.copy()

    ucomp, vcomp = read_grib_data(wind, dates)
    for date in dates:
        #         min_temp = []
        #         max_temp = []
        #         mean_temp = []
        if not date in ucomp:
            continue
        dom_dir, dom_vel, res_max, dir_max = extract_arrays(ucomp[date], vcomp[date])
        #    temp_to_tif(min_temp,'minimum')
        #   temp_to_tif(max_temp,'maximum')
        #   temp_to_tif(mean_temp,'mean')
        try:
            wind_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_wind[dataset_np_wind[:, 2] == date], res_max, dir_max, dom_vel,dom_dir)
            for array in wind_arr:
                my_list.append(array.tolist())
            print(date,'ok')
        except:
            print(date,'failed')
            continue

print('DONE in %s minutes'%((time.time() - abs_time)/60))

with open("/home/sg/Projects/FIrehub-model/no_fire_90/no_fire_wind.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(my_list)

i=1
#centr_msg_date_wgs['firedate'] = centr_msg_date_wgs['firedate'].astype(str)
#centr_msg_date_wgs.to_file('/home/sg/Projects/FIrehub-model/Dataset_1st/non fire pixels/no_fire_centr_wind.shp')
#centr_msg_date_wgs_upd.dtypes

