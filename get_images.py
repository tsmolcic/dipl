# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:13:32 2018

@author: Tomislav
"""

import os
import requests

cwd = os.getcwd()
raw_data_dir = cwd+'/sdss_data/raw_imgs'

if not os.path.exists(raw_data_dir):
    print('creating directories...\n')
    os.makedirs(raw_data_dir)

num_of_imgs = 2 #reduce if needed

start_ind = 100
end_ind = start_ind + num_of_imgs #max(end_ind)=380

count = 1

for i in range(start_ind,end_ind):
    print('downloading image '+str(count)+'/'+str(num_of_imgs)+'...')
    url = 'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/211/4/frame-irg-000211-4-0'+str(i)+'.jpg'
    r = requests.get(url, allow_redirects=True)
    file_name = 'img'+str(count)+'.jpg'
    open(raw_data_dir+'/'+file_name, 'wb').write(r.content)
    count += 1
    