import numpy as np
from osgeo import gdal, ogr, osr
import os
import csv
import sys
from statistics import mode

stat_algorithm = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min,
    'majority': np.bincount
}

srs = osr.SpatialReference()
srs.ImportFromEPSG(32634)

ras = gdal.Open('/mnt/storageapplications/data/Floodhub+/mandra_RUN2022/GIS/dem/dem2_forslopes&elevation/moWGS84.tif')
gt = ras.GetGeoTransform()
inv_gt = gdal.InvGeoTransform(gt)

total = []
first = []
second = []
i = 0

# for nband in range(ras.RasterCount):
#     k_meta = 'Band_%d' % (nband + 1)
#     band_name = ras2.GetRasterBand(nband + 1).GetMetadata_Dict()[k_meta]
#     band_name = band_name.split('.')[0]
#     print band_name

ws_tmp = '/mnt/storageapplications/data/Floodhub+/mandra_RUN2022/GIS/subbasins/'
outname = 'dem_mean_32634.csv'
out_csv = os.path.join(ws_tmp, outname)


for band in range(ras.RasterCount):
    #k_meta = 'Band_%d' % (band + 1)
    #band_name = 'lu_tif'


    values = ras.GetRasterBand(band + 1).ReadAsArray().astype(np.float32)
    print(values)
    i += 1

    k_meta = 'Band_%d' % (band + 1)
    band_name = 'dem'
    band_name = band_name.split('.')[0]
    #bands1 = band_name.split('_')[-1]
    #band_name = band_name.split('_')[4][:8] + '_' + bands1
    print(band_name)

    cnt = 1
    col_list = []
    col_list.append(band_name)
    first.append("id")
    #second.append("code")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open('/mnt/storageapplications/data/Floodhub+/mandra_RUN2022/GIS/subbasins/subbasins2020newcn_F.shp', 0)
    layer = dataSource.GetLayer()

    for feature in layer:
        geom = feature.GetGeometryRef()
        geom = geom.ExportToWkt()
        id = feature.GetField('id')
        # code = feature.GetField("code")
        # if not(code==1 or code==0):
        #     continue
        # code = 1
        if i == 1:
            first.append(id)
            # second.append(code)
        parcel_data = [id]
        try:


            vect_tmp_drv2 = ogr.GetDriverByName('MEMORY')
            vect_tmp_src2 = vect_tmp_drv2.CreateDataSource('')
            vect_tmp_lyr2 = vect_tmp_src2.CreateLayer('', srs, ogr.wkbPolygon)
            vect_tmp_lyr2.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

            feat2 = ogr.Feature(vect_tmp_lyr2.GetLayerDefn())
            feat2.SetField("id", id)
            feat_geom2 = ogr.CreateGeometryFromWkt(geom)
            feat2.SetGeometry(feat_geom2)
            vect_tmp_lyr2.CreateFeature(feat2)

            xmin, xmax, ymin, ymax = feat_geom2.GetEnvelope()

            off_ulx2, off_uly2 = map(int, gdal.ApplyGeoTransform(inv_gt, xmin, ymax))
            off_lrx2, off_lry2 = map(int, gdal.ApplyGeoTransform(inv_gt, xmax, ymin))
            rows2, columns2 = (off_lry2 - off_uly2) + 1, (off_lrx2 - off_ulx2) + 1
            ras_tmp = gdal.GetDriverByName('MEM').Create('', columns2, rows2, 1, gdal.GDT_Byte)
            ras_tmp.SetProjection(ras.GetProjection())
            ras_gt = list(gt)
            ras_gt[0], ras_gt[3] = gdal.ApplyGeoTransform(gt, off_ulx2, off_uly2)
            ras_tmp.SetGeoTransform(ras_gt)

            gdal.RasterizeLayer(ras_tmp, [1], vect_tmp_lyr2, burn_values=[1])
            mask = ras_tmp.GetRasterBand(1).ReadAsArray()
            aa = off_uly2
            bb = off_lry2 + 1
            cc = off_ulx2
            dd = off_lrx2 + 1


            # zone_ras_init = np.ma.masked_array(result[aa:bb, cc:dd], np.logical_not(mask), fill_value=-999999)
            zone_ras =  np.ma.masked_array(values[aa:bb, cc:dd])# np.ma.masked_array(values[aa:bb, cc:dd], np.logical_not(mask), fill_value=-999999)
            zone_ras_list2 = zone_ras.compressed()#.tolist()

            if False:
                # col_list.append(-999999)
                col_list.append(-999999)
            else:

                if np.all(zone_ras_list2 == -999999):
                    x = -999999
                else:
                    #x = mode(zone_ras_list2)
                #HERE WE CHANGE (nanmax,nanmean,nanmin)
                    x = np.nanmean(zone_ras_list2)
                col_list.append(x)
                # print x
                # print np.nanmean(zone_ras_list2)
            #' #''
            #if perc > 0.8:
                ##col_list.append(-999999)
                #zone_ras_list2[zone_ras_list2 == -999999] = np.nan
                #if (np.isnan(zone_ras_list2).all()):
                    #col_list.append = -999999
                #else:
                    #col_list.append(np.nanmean(zone_ras_list2))
            #else:
                #zone_ras_list2[zone_ras_list2 == -999999] = np.nan
                #col_list.append(np.nanmean(zone_ras_list2))
                #print np.nanmean(zone_ras_list2)
            #' ''
            del zone_ras_list2
            del zone_ras
        except Exception as e:
            print(e)
            col_list.append(-999999)
            cnt += 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            #logger.info('Line: %d,\t Type: %s,\t Message: %s', exc_tb.tb_lineno, exc_type, exc_obj)
            continue

            # if exists==1:
            # parcel_data.append(family)
            # parcel_data.append(season)
    if i == 1:
        total.append(first)
        # total.append(second)
    total.append(col_list)

    del values

rows = zip(*total)
with open(out_csv, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        # print row
        writer.writerow(row)