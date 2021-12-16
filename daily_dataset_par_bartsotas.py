import time
import pygrib
import multiprocessing as mp
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import ndvi_bartsotas
import temperature_daily_bartsotas
import precipitation_daily_bartsotas
import wind_daily_bartsotas
import geopandas as gpd
from functools import reduce
import datetime
from os import chdir

def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def parallel_process(date,meteo_data):
    abs_time = time.time()
    dataset_join = gpd.read_file('/home/sg/Projects/FIrehub-model/dataset_greece_static/greece_lc_dem_withtiles/greece_withtiles.shp')
    #dataset_join['x'] = dataset_join.geometry.x
    #dataset_join['y'] = dataset_join.geometry.y

    print(mp.cpu_count())

    start_time = time.time()

    #ndvi = ndvi_23072018.run_ndvi(date)
    procdict={}
    #q=mp.Queue()
    #q2=mp.Queue()
    queues = [mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()]
    #p =  mp.Process(target=testpar, args=(14,q,))
    #p2 =  mp.Process(target=testpar, args=(20,q2,))
    #procdict['p'] = {'proc':p, 'start':datetime.now(), 'status':'running', 'queue':q}
    #procdict['p2'] = {'proc':p2, 'start':datetime.now(), 'status':'running', 'queue':q2}

    procdict['ndvi_process'] = {'proc':mp.Process(target=ndvi_bartsotas.run_ndvi, args=[dataset_join,queues[0]]), 'start':datetime.datetime.now(), 'status':'running', 'queue':queues[0]}
    procdict['temp_process'] = {'proc':mp.Process(target=temperature_daily_bartsotas.run_temp, args=[dataset_join,meteo_data,queues[1]]), 'start':datetime.datetime.now(), 'status':'running', 'queue':queues[1]}
    procdict['rain_process'] = {'proc':mp.Process(target=precipitation_daily_bartsotas.run_prec, args=[dataset_join,queues[2]]), 'start':datetime.datetime.now(),'status':'running', 'queue':queues[2]}
    procdict['wind_process'] = {'proc':mp.Process(target=wind_daily_bartsotas.run_wind, args=[dataset_join,meteo_data,queues[3]]), 'start':datetime.datetime.now(),'status':'running', 'queue':queues[3]}

    for p in procdict:
        procdict[p]['proc'].start()

    print('Running processes...')
    dfdict = {}
    while True:
        for proc in procdict:
            try:
                while not procdict[proc]['queue'].empty():
                    dfdict[proc]=procdict[proc]['queue'].get_nowait()
                    print(proc)
                    print(dfdict[proc])
            except:
                i=1
        for proc in procdict:
            if not procdict[proc]['proc'].is_alive() and procdict[proc]['status']=='running':
                print('Process %s finished in %s'%(proc,datetime.datetime.now()-procdict[proc]['start']))
                procdict[proc]['status']='finished'
                break
        '''
        if not procdict[proc]['proc'].is_alive():
            del procdict[proc]
        '''
        if all(not procdict[x]['proc'].is_alive() for x in procdict):
            dfs=[dfdict[proc] for proc in procdict]
            #dfs = dfdict['temp_process']
            #dfs = [dfdict['ndvi_process'], dfdict['temp_process'], dfdict['rain_process'], dfdict['wind_process']]
            df_final = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)
            df_final.to_csv('/home/sg/Projects/FIrehub-model/Bartsotas/dataset_07072019/'+ date_str+'.csv')
            print("Script finished in %s minutes" %((time.time() - abs_time)/60))
            break
        time.sleep(1)



date = (date.today() - timedelta(days=1))
date_str = date.strftime("%Y%m%d")
meteo = 'WRF-'+date_str+'.grb2'
chdir('/home/sg/test_modis_download')
meteo_data = pygrib.open(meteo)

parallel_process(date,meteo_data)