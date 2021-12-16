import time

abs_time = time.time()
date = '20180723'

start_time = time.time()
import ndvi_23072018
ndvi = ndvi_23072018.run_ndvi(date)
print('NDVI DONE in %s minutes' % ((time.time() - start_time)/60))

start_time = time.time()
import temperature_daily_numpy
temp = temperature_daily_numpy.run_temp(date)
print('Temperature DONE in %s minutes' % ((time.time() - start_time)/60))

start_time = time.time()
import precipitation_daily_numpy
prec = precipitation_daily_numpy.run_prec(date)
print('Precipitation DONE in %s minutes' % ((time.time() - start_time)/60))


start_time = time.time()
import wind_daily_numpy
wind = wind_daily_numpy.run_wind(date)
print('Wind DONE in %s minutes' % ((time.time() - start_time)/60))

print('All DONE in %s minutes' % ((time.time() - abs_time)/60))

print(precipitation_arr)