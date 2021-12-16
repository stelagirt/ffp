import pandas as pd
import sys
import os
#from sqlalchemy import create_engine
#import psycopg2
import numpy as np
import calendar
from os import listdir
from os.path import isfile, join

def fill_nulls(df,column):
    print(column,len(df[(df[column] =='--')|(df[column] =='-1000')|(df[column] ==-1000)|(df[column].isnull())].index))
    mean = df[(df[column] !='--')&(df[column] !='-1000')&(df[column] !=-1000)][column].astype('float').mean()
    df.loc[(df[column] =='--')|(df[column] =='-1000')|(df[column] ==-1000)|(df[column].isnull()),column]=mean
    df[column] = df[column].astype('float')

#year = 2011
#month = '04'
year = sys.argv[1]
month = sys.argv[2]
start = str(year)+str(month)

path = '/users/pa21/sgirtsou/production/'+str(year)+'/'+str(month)
print(path)
os.chdir(path)

start = str(year)+str(month)
month_str = calendar.month_name[int(month)]

fires=pd.read_csv('/users/pa21/sgirtsou/transfered_files/data/bsm_2020_cells.csv')

columns = ['ndvi', 'evi', 'max_temp', 'mean_temp','min_temp', 'prcp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir',
       'lst_day', 'lst_night', 'max_dew_temp', 'mean_dew_temp', 'min_dew_temp']

files = list(f for f in os.listdir(path) if f.startswith(start))

filename = (month_str+'_'+str(year)+'_dataset.csv').lower()
print('Making '+ filename +' file')

merge = []
for i,file in enumerate(files):
    print(i,file)
    df = pd.read_csv(file,low_memory=False)
    for column in columns:
        fill_nulls(df,column)
    df['firedate'] = file[0:8]
    df['fire'] = 0
    if i == 0:
        df.to_csv(filename,index = False)
    else:
        df.to_csv(filename, index= False, mode='a', header=False)
    print(file+'_ok')

filename_2 = (month_str+'_'+str(year)+'_dummies.csv').lower()
print('Making '+filename_2+' file')

chunksize = 10 ** 6
count = 0
for i,chunk in enumerate(pd.read_csv(filename, chunksize=chunksize,low_memory=False)):
   #chunk['corine_2'] = (chunk['Corine']/10).astype('int')
    chunk = chunk[chunk.dom_dir != '--']
    chunk = chunk[chunk.max_temp != '--']
    chunk = chunk[chunk.ndvi!=-1000]
    chunk['dom_dir'] = chunk.dom_dir.astype('float').astype('int')
    chunk = chunk[(chunk.dom_dir >-100) & (chunk.dom_dir <100)]
    chunk['dir_max'] = chunk.dir_max.astype('float').astype('int')
    chunk = chunk.rename(columns = {'DEM':'dem','Corine':'corine','Slope':'slope', 'Aspect':'aspect', 'Curvature':
                                        'curvature','ndvi':'ndvi_new','prcp':'rain_7days', 'res_max':'dom_vel','dom_vel':'res_max'})
        
    chunk = chunk[['id','firedate','max_temp', 'min_temp', 'mean_temp','res_max','dom_vel','rain_7days',
                        'dem','slope','curvature','aspect','corine','dir_max', 'dom_dir', 'ndvi_new',
                       'evi','lst_day', 'lst_night', 'max_dew_temp', 'mean_dew_temp', 'min_dew_temp',
                   'fire' ]]
    chunk = pd.get_dummies(data=chunk, columns=['dir_max','dom_dir','corine'])
    for index,row in fires.iterrows():
        chunk.loc[(chunk.id == row.id) & (chunk.firedate == row.firedate_g), 'fire'] = 1
    if i == 0:
        chunk.to_csv(filename_2,index = False)
    else:
        chunk.to_csv(filename_2, index= False, mode='a', header=False)
    c = chunk[['fire']].sum()
    count += c
    print('Number of fires')
    print(c)
print(count)
