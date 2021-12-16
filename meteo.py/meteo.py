import requests
    
with open('meteo.txt','a') as fl:
    for year in range(2013, 2020):
        for month in range(5,10):
            r = requests.get('http://meteosearch.meteo.gr/data/portorafti/{}-{}.txt'.format(str(year+1), str(month+1).zfill(2)))
            if r.status_code == 200:
                fl.write(r.content.decode("ISO-8859-1"))