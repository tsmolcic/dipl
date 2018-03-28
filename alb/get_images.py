# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:13:32 2018

@author: Tomislav
"""

import os
import requests

num_of_imgs = 1 #reduce if needed

cwd = os.getcwd()

urls = ['https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/109/4/frame-irg-000109-4-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/109/1/frame-irg-000109-1-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/211/1/frame-irg-000211-1-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/211/1/frame-irg-000211-2-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/211/1/frame-irg-000211-3-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/211/1/frame-irg-000211-6-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-1-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-2-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-3-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-4-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-5-0',
        'https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/273/1/frame-irg-000273-6-0']

dirs  = []

print('creating directories...\n')
for i in range(len(urls)):
    dir_name = cwd+'/sdss_data/raw_imgs_'+str(i)+'/'
    dirs.append(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

start_ind = 100
end_ind = start_ind + num_of_imgs

for j in range(len(urls)):
    count = 1
    url_start = urls[j]
    for i in range(start_ind,end_ind):
        print('downloading image '+str(count)+'/'+str(num_of_imgs)+'...')
        url_path = url_start+str(i)+'.jpg'
        try:
            r = requests.get(url_path, allow_redirects=True)
            file_name = 'u_'+str(j)+'_img'+str(count)+'.jpg'
            open(dirs[j]+'/'+file_name, 'wb').write(r.content)
            count += 1
        except Exception as e:
            print('Unable to fetch url - skipping.')