from pymodis import downmodis
import os
import json
from datetime import date, timedelta

def download_last_modis():
    today = date.today()
    start = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (today - timedelta(days=5)).strftime("%Y-%m-%d")

    # tiles to download-Greece region
    tiles = 'h19v05,h20v05,h19v04'

    # starting day
    day = start
    # finishing day
    enddate = end
    # username
    user = 'stelgirt'
    # password
    password = 'Kamilia90'
    # destination folder
    dest = '/home/sg/test_modis_download/last'
    # log
    delta = 1
    product = 'MYD13A1.006'  # ,MOD13A1.006'
    path = 'MOLA'  # ,MOLT'
    # product2 = 'MOD13A1.006'
    # path2 = 'MOLT'


    modisDown = downmodis.downModis(destinationFolder=dest, tiles=tiles, today=day, enddate=enddate,
                                    delta=delta, user=user, password=password, product=product, path=path,
                                    jpg=False, debug=False, timeout=30, checkgdal=True)

    modisDown.connect()
    #files_list = modisDown.getFilesList
    #new_files = modisDown.getAllDays()
    modisDown.downloadsAllDay()
    modisDown.removeEmptyFiles()

    my_path = '/home/sg/test_modis_download/last'
    with open("/home/sg/test_modis_download/archive/most_recent.json") as f:
        last_images = json.load(f)
    i = 0
    for image in os.listdir(my_path):
        if image.endswith('hdf'):
            i = 1
            check = modisDown.checkFile(image)
            if check == 1:
                os.rename("/home/sg/test_modis_download/last/" + image, "/home/sg/test_modis_download/archive/" + image)
                os.rename("/home/sg/test_modis_download/last/" + image + '.xml',
                          "/home/sg/test_modis_download/archive/" + image + '.xml')
                tile = image[17:23]
                last_images[tile] = image
            else:
                print('Image %s corrupted. Keeping previous image' % image)
    if i == 0:
        print('MOLA: No new images found')
        product = 'MOD13A1.006'
        path = 'MOLT'

        modisDown = downmodis.downModis(destinationFolder=dest, tiles=tiles, today=day, enddate=enddate,
                                        delta=delta, user=user, password=password, product=product, path=path,
                                        jpg=False, debug=False, timeout=30, checkgdal=True)
        modisDown.connect()
        modisDown.downloadsAllDay()
        modisDown.removeEmptyFiles()

        for image in os.listdir(my_path):
            if image.endswith('hdf'):
                i = 1
                check = modisDown.checkFile(image)
                if check == 1:
                    os.rename("/home/sg/test_modis_download/last/" + image, "/home/sg/test_modis_download/archive/" + image)
                    os.rename("/home/sg/test_modis_download/last/" + image + '.xml',
                              "/home/sg/test_modis_download/archive/" + image + '.xml')
                    tile = image[17:23]
                    last_images[tile] = image
                else:
                    print('Image %s corrupted. Keeping previous image' % image)

    if i == 0:
        print('MOLT: No new images found')


    json_images = json.dumps(last_images)
    f = open("/home/sg/test_modis_download/archive/most_recent.json", "w")
    f.write(json_images)
    f.close()

download_last_modis()

