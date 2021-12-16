import pandas as pd
import geopandas as gpd
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
import datetime as dt

#cells = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_cells/cells_masked_greek.shp')

#cells = cells.drop(columns = ['the_geom', 'the_geom_p', 'the_geom_1','the_geom_c', 'the_geom_2', 'the_geom_m', 'the_geom_s', 'the_geom_3'])

centoids = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')
cells_id = pd.read_csv('/home/sg/Projects/FIrehub-model/tests_2019/August_2019/fires_2019_in_cells/fires_2019_cells_id_date.csv')

cells_id.columns = ['OBJECTID', 'fire_id', 'firedate', 'id', 'gid']
cells_id['id'] = cells_id.id.astype('float64')

cells_id_date = cells_id[['firedate','id']]
#cells_id_date = cells_id_date.groupby(by = 'firedate', axis =1)

cells_dict = {k: list(v) for k, v in cells_id_date.groupby('firedate')['id']}

id_list = cells_id.id.tolist()

mypath = '/home/sg/Projects/FIrehub-model/tests_2019/June_2019/daily_csvs'
os.chdir('/home/sg/Projects/FIrehub-model/tests_2019/June_2019/daily_csvs')
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


centoids['fire2019'] = 0
centoids['fire2019'].loc[centoids['id'].apply(lambda x: x in id_list)]= centoids['fire2019'].loc[centoids['id'].apply(lambda x: x in id_list)] +1

for file in onlyfiles:
    if file.endswith('csv'):
        date_string = file[0:8]
        date_dt = datetime.strptime(date_string, '%Y%m%d')
        date_str = datetime.strftime(date_dt, '%d/%m/%Y')
        if date_str in cells_dict.keys():
            df = pd.read_csv(file)
            df['fire'] = 0
            df['fire'].loc[df['id'].apply(lambda x: x in cells_dict[date_str])] = df['fire'].loc[df['id'].apply(lambda x: x in cells_dict[date_str])]+1
            df = df[['id', 'Code_18','DEMGreece_', 'Slope_DEMG', 'Aspect_DEM', 'Curvatu_DE','ndvi', 'max_temp', 'min_temp',
           'mean_temp', 'prcp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','x', 'y','fire']]
            df.columns = ['id', 'Corine','DEM', 'Slope', 'Aspect', 'Curvature','ndvi', 'max_temp', 'min_temp',
           'mean_temp', 'rain_7days', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','x', 'y','fire']
            df = df[df.max_temp != '--']
            df.to_csv('/home/sg/Projects/FIrehub-model/tests_2019/June_2019/csvs_withfire/fire' + date_string + '.csv')
        elif date_str not in cells_dict.keys():
            df = pd.read_csv(file)
            df['fire'] = 0
            df = df[['id', 'Code_18', 'DEMGreece_', 'Slope_DEMG', 'Aspect_DEM', 'Curvatu_DE', 'ndvi', 'max_temp','min_temp',
            'mean_temp', 'prcp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'x', 'y', 'fire']]
            df.columns = ['id', 'Corine', 'DEM', 'Slope', 'Aspect', 'Curvature', 'ndvi', 'max_temp', 'min_temp','mean_temp', 'rain_7days', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'x', 'y', 'fire']
            df = df[df.max_temp != '--']
            df.to_csv('/home/sg/Projects/FIrehub-model/tests_2019/June_2019/csvs_withoutfire/no_fire' + date_string + '.csv')
            print(date_str)

#fires2019 = centoids[centoids.fire2019 == 1]
#fires2019.to_file('/home/sg/Projects/FIrehub-model/August_2019/fires_2019_in_cells/fires_2019_centroids')


i=2