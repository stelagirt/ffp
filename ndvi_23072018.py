import pandas as pd
import datetime
from datetime import datetime as dt
import os
from osgeo import osr
from osgeo import gdal, ogr, gdalconst
import numpy as np
import numpy.ma as ma
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import time
from shapely import geometry


def find_img(y, x,hdf_dict):  # x=firedate, y=row
    # print(x.year, x)
    new_dict = [k for k, v in hdf_dict.items() if (dt.strptime(k, '%Y%m%d') <= x) and k.startswith(str(x.year))]
    # print(new_dict)
    try:
        min_date = new_dict[0]
        min_days = x - dt.strptime(new_dict[0], '%Y%m%d')
        for item in new_dict:
            if x - dt.strptime(item, '%Y%m%d') < min_days:
                min_days = x - dt.strptime(item, '%Y%m%d')
                min_date = item
        image_name = list(filter(lambda a: a.split(".")[2] == y[9], hdf_dict[min_date]))
        return (image_name[0])
    except:
        print("Exception occured")

def nearest(data, min_date_str, hdf_dict,M):
    image_name = list(filter(lambda a: a.split(".")[2] == data[9], hdf_dict[min_date_str]))
    data[M] = image_name[0]
    return(data)


def import_hdf(hdf):
    ds = gdal.Open(hdf)
    t_srs = osr.SpatialReference()
    t_srs.ImportFromEPSG(4326)
    src_ds = gdal.Open(ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
    src_ds_evi = gdal.Open(ds.GetSubDatasets()[1][0], gdal.GA_ReadOnly)
    dst_wkt = t_srs.ExportToWkt()
    dswrap = gdal.Warp('', src_ds, dstSRS='EPSG:4326', outputType=gdal.GDT_Int16, format='VRT')
    dswrap_evi = gdal.Warp('', src_ds_evi, dstSRS='EPSG:4326', outputType=gdal.GDT_Int16, format='VRT')
    xmin = dswrap.GetGeoTransform()[0]
    ymax = dswrap.GetGeoTransform()[3]
    pixel_x = abs(dswrap.GetGeoTransform()[1])
    pixel_y = abs(dswrap.GetGeoTransform()[5])
    hdf_ar = dswrap.ReadAsArray().astype(np.float)
    hdf_ar_evi = dswrap_evi.ReadAsArray().astype(np.float)
    data_np = np.array(hdf_ar)
    data_np_evi = np.array(hdf_ar_evi)
    data_m = ma.masked_where(data_np == -3000, data_np)  # Added by Stella
    data_m_evi = ma.masked_where(data_np_evi == -3000, data_np_evi)
    data_masked = data_m * 0.0001  # Added by Stella
    data_masked_evi = data_m_evi * 0.0001  # Added by Stella
    return xmin, ymax, pixel_x, pixel_y, data_masked, data_masked_evi

def load_hdf(fname, layer_dict, layer):
    print("\nLoading %s" % os.path.split(fname)[1])
    # convenience lookup so we can use short names
    layer_key = layer_dict.get(layer)
    hdf = gdal.Open(fname)
    sdsdict = hdf.GetMetadata('SUBDATASETS')
    sdslist =[sdsdict[k] for k in sdsdict.keys() if '_NAME' in k]
    sds = []
    for n in sdslist:
        sds.append(gdal.Open(n))
    # make sure the layer we want is in the file
    if layer_key:
        full_layer = [i for i in sdslist if layer_key in i] # returns empty if not found
        idx = sdslist.index(full_layer[0])
        data = sds[idx].ReadAsArray()
        data_np = np.array(data) #Added by Stella
        data_m = ma.masked_where(data_np == -3000, data_np) #Added by Stella
        data_masked = data_m*0.0001 #Added by Stella
        #print_data(layer, data)
        a = 'GRINGPOINTLATITUDE.1'
        b = 'GRINGPOINTLONGITUDE.1'
        c = hdf.GetMetadata()['LOCALGRANULEID'].split('.')[2]
        p1 = geometry.Point(float(i.GetMetadata()[b].split(",")[0]),float(i.GetMetadata()[a].split(",")[0]))
        p2 = geometry.Point(float(i.GetMetadata()[b].split(",")[1]),float(i.GetMetadata()[a].split(",")[1]))
        p3 = geometry.Point(float(i.GetMetadata()[b].split(",")[2]),float(i.GetMetadata()[a].split(",")[2]))
        p4 = geometry.Point(float(i.GetMetadata()[b].split(",")[3]),float(i.GetMetadata()[a].split(",")[3]))
        x_min = p1.x
        print("x_min%s" %x_min)
        y_max = p2.y
        print('y_max%s' %y_max)
        pixel_x = p1.distance(p4)/2400
        print('pixel_x%s' %pixel_x)
        pixel_y = p1.distance(p2)/2400
        print('pixel_y%s' %pixel_y)
        #xmin = float(hdf.GetMetadata()['WESTBOUNDINGCOORDINATE'])
        #ymax = float(hdf.GetMetadata()['NORTHBOUNDINGCOORDINATE'])
        #pixel_x = (float(hdf.GetMetadata()['EASTBOUNDINGCOORDINATE']) - float(hdf.GetMetadata()['WESTBOUNDINGCOORDINATE']))/2400
        #pixel_y = (float(hdf.GetMetadata()['NORTHBOUNDINGCOORDINATE']) - float(hdf.GetMetadata()['SOUTHBOUNDINGCOORDINATE']))/2400
        return(data_masked, x_min, y_max, pixel_x, pixel_y)
    else:
        print("Layer %s not found" % layer)

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


def attribute_geodataframe_numpy(x, xmin, ymax, pixel_x, pixel_y, M,ndvi,evi):
    x_pos, y_pos = take_point_position(xmin, ymax, pixel_x, pixel_y, x[11], x[12])
    ndvi_values = -1000
    evi_values = -1000
    # print('Before%s'%x[28])
    if x_pos > ndvi.shape[1] - 1 or y_pos > ndvi.shape[0] - 1:
        return
        # print(x_pos,y_pos)
    # if x_pos > ndvi.shape[1]-1 or y_pos > ndvi.shape[0]-1:
    if ndvi[y_pos][x_pos]:
        ndvi_values = ndvi[y_pos][x_pos]
        evi_values = evi[y_pos][x_pos]
    else:
        for i in range(1, 5):
            if not ndvi[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= ndvi.shape[1] - i and y_pos <= \
                    ndvi.shape[0] - i:
                ndvi_values = np.mean(ndvi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                evi_values = np.mean(evi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
                if ndvi_values != -1000:
                    break
    x[M] = ndvi_values
    x[M+1] = evi_values
    return (x)


def attribute_filtered_numpy(ndvi, evi,xmin, ymax, pixel_x, pixel_y):
    ndvi_list = []
    evi_list = []
    print(x[22])
    x_pos, y_pos = take_point_position(xmin, ymax, pixel_x, pixel_y, x[11], x[12])
    print(x_pos, y_pos)
    if x_pos > 2400 or y_pos > 2400:
        print('Outlier')

    for i in range(1, 5):
        if not ndvi[y_pos][x_pos] and x_pos >= i and y_pos >= i and x_pos <= 2400 - i and y_pos <= 2400 - i:
            ndvi_values = np.mean(ndvi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
            evi_values = np.mean(evi[(y_pos - i):(y_pos + i + 1), (x_pos - i):(x_pos + i + 1)])
            if ndvi_values:
                break
    else:
        ndvi_values = ndvi[y_pos][x_pos]
        evi_values = evi[y_pos][x_pos]
    if ndvi_values:
        ndvi_list.append(ndvi_values)
        evi_list.append(evi_values)
    # i[27] = ndvi_values
    # i[28] = evi_values
    # cnt+=1
    # if cnt%10000 == 0:
    # print("Elapsed_time: {}".format(time.time() - start_time))
    # print(cnt)
    return (ndvi_list, evi_list)


def image_in_greece(tile):
    image = ''
    if tile == 'h19v04':
        image = 'MYD13A1.A2018201.h19v04.006.2018220194332.hdf'
    elif tile == 'h19v05':
        image = 'MYD13A1.A2018201.h19v05.006.2018220194230.hdf'
    elif tile == 'h20v05':
        image = 'MYD13A1.A2018201.h20v05.006.2018220194217.hdf'
    return (image)


def run_ndvi(dataset_join,fire_date,q):
    #dataset_join is the Greek points with the tile info for each point
    #dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')
    tiles = list(filter(None, list(dataset_join.tile.unique())))

    dataset_np = dataset_join.to_numpy()
    N, M = dataset_np.shape
    dataset_np_ndvi = np.hstack((dataset_np, np.zeros((dataset_np.shape[0], 2))))

    #fire_date = '20190801'
    fire_date_dt = dt.strptime(fire_date,'%Y%m%d')
    my_path = "/users/pa21/sgirtsou/transfered_files/data/ndvi_2020/"

    hdf_name = []
    for image in os.listdir(my_path):
        if image.endswith('hdf') and str(fire_date_dt.year) in image:
            hdf_name.append(image)
    hdf_date = []
    for image in hdf_name:
        date = image.split(".")[1][1:8]
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

    my_list = []
    for tile in tiles:
        print(tile)
        ndvi_arr = np.apply_along_axis(nearest, 1, dataset_np_ndvi[dataset_np_ndvi[:, 9] == tile],min_date_str,hdf_dict,M)
        for array in ndvi_arr:
            my_list.append(array.tolist())

    dataset_np_ndvi = np.asarray(my_list)
    images = list(np.unique(dataset_np_ndvi[:, M]))

    #optimization with numpy
    os.chdir('/users/pa21/sgirtsou/transfered_files/data/ndvi_2020')
    # basket = np.array([])
    # basket = np.empty((0, dataset_np.shape[1] + 1))

    start_time = time.time()

    i = 0
    my_list = []
    for image in images:
        print(i, image)
        xmin, ymax, pixel_x, pixel_y, ndvi, evi = import_hdf(image)
        # temp_to_tif(xmin, ymax, pixel_x, pixel_y, ndvi, image)
        ndvi_arr = np.apply_along_axis(attribute_geodataframe_numpy, 1, dataset_np_ndvi[dataset_np_ndvi[:, M] == image],
                                       xmin, ymax,
                                       pixel_x, pixel_y,M, ndvi,evi)
        for array in ndvi_arr:
            my_list.append(array.tolist())
            # basket = np.append(basket, [array], axis=0)
        i += 1

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(elapsed_time)
        # basket = basket.filled(np.nan)
        # basket = np.append(basket,temp)
    #dataset_join['image'] = -1000
    dataset_join['ndvi'] = -1000
    dataset_join['evi'] = -1000
    ndvi_df = gpd.GeoDataFrame(my_list)  # columns=dataset_join.columns)
    ndvi_df.columns = dataset_join.columns
    ndvi_df['id'] = ndvi_df.id.astype('float')
    # ndvi_df = ndvi_df.groupby('id').max().reset_index()
    ndvi_df = ndvi_df.dropna()
    ndvi_df = ndvi_df[ndvi_df.ndvi !=-1000]
    # ndvi_df.to_csv('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_id_geom/greece_ndvi.csv')
    q.put(ndvi_df)
