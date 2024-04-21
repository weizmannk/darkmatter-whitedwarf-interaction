
import pandas as pd
from astropy.table import Table
import glob
import os
from tqdm.auto import tqdm
import urllib
import requests



api_token = "1f9d7756-7fe0-4c29-ab52-8c5b51a1f8da"
host = "https://fritz.science/"

targetdir ='../../ZTF'
datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/../data/lc_gap_csv"

outdir = targetdir           
if not os.path.isdir(outdir):
    os.makedirs(outdir)


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """
    
    return glob.glob(datapath+'/**/*.csv', recursive=True)


classification = []
source_name =[]
ob_id = []
prob = []

ra = []
dec = []
redshift = []

Fritz_link = []

s =0
Ia_number= []
ztf_files = data_filename(datapath)

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
    
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]

        endpoint = f"sources/{name}"
        url = urllib.parse.urljoin(host, f'/api/{endpoint}')
        headers = {'Authorization': f'token {api_token}'}

        r = requests.get(url, headers=headers)
        data_out = r.json()['data']
                  
        if 'classifications' in data_out  and  len(data_out['classifications']) != 0:

            print(data_out)
            source_name.append(name)
            
            ra.append(data_out['ra'])
            dec.append(data_out['dec'])
            redshift.append(data_out['redshift'])
            
            ob_id.append(data_out['classifications'][0]['obj_id'])
            classification.append(data_out['classifications'][0]['classification'])

            prob.append(data_out['classifications'][0]['probability'])
         
            if name == data_out['classifications'][0]['obj_id']:
                URL = urllib.parse.urljoin(host, f'source/{name}')
                Fritz_link.append(URL)
            else : 
                Fritz_link.append(f"{name} != {ob_id}")
                s+=1
                
            #if 'Ia' in data_out['classifications'][0]['classification']:
            #    Ia_number.append(data_out['classifications'][0]['classification'])
            
            print(data_out)

            
            
            del name, endpoint, url, headers, r, data_out, URL
        progress.update()
            
            
        
    print("\n===========================================\n")
    print (f"We have {s} which present some difference between the name and Ob-id\n")
    
    
    print("=========================================================================\n")
    print(f"The number of Objetcs classified as a Ia supernovae is :  {len(Ia_number)}\n")
    print("=========================================================================\n")
    
    table_transient = Table({'source_name':source_name, 'classification': classification, 'probability':prob, 'ra':ra, 'dec':dec, 'redshift': redshift, 'fritz_link':Fritz_link})
    
    table_transient.write(f"{outdir}/ZTF_Gap_Trasient_unclassifications.dat", format='ascii.tab', overwrite=True)
            
        

        
        
            
