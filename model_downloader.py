import requests
from tqdm import tqdm
from urllib.parse import urlparse
import os

download_dir = 'D:\Python Notebooks\R&D - Image Super Resolution\models'
#models = ['EDSR', 'ESPCN', 'FSRCNN', 'LapSRN']
models = {
    'EDSR':'D:\Python Notebooks\R&D - Image Super Resolution\models\edsr.txt',
    'FSRCNN':'D:\Python Notebooks\R&D - Image Super Resolution\models\fsrcnn.txt',
    'LapSRN':'D:\Python Notebooks\R&D - Image Super Resolution\models\lapsrn.txt', 
    'ESPCN':'D:\Python Notebooks\R&D - Image Super Resolution\models\espcn.txt'
}
def gen_folders(download_dir, models):
    for m in models.keys():
        path = os.path.join(download_dir, m)
        isExist = os.path.exists(path)
        if not isExist: 
            os.mkdir(path)

def download_files(links, path):
    for url in links:
        r = requests.get(url, stream=True)
        total_size= int(r.headers.get('content-length', 0))
        block = 1024 #1 Kibibyte
        
        a = urlparse(url)
        final_path = os.path.join(path, os.path.basename(a.path))      
        print(f'=====\n\nDownloading {os.path.basename(a.path)}')
  
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(final_path, 'wb') as file:
            for data in r.iter_content(block):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")


def parse_links(model, file):
    f = open(file, 'r')
    links = f.readlines()
    f.close()
    path = os.path.join(download_dir, model)
    download_files(links, path)

gen_folders(download_dir, models) 

for k,v in models.items():
    parse_links(k, v)
    print(f'Done Downloading {k} models')