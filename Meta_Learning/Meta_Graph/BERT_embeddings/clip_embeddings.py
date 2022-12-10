# Code to clip Google Review sentence embeddings

import numpy as np 
from os import listdir
from os.path import isfile, join

file_root = "data/CITIES/emb"
city_files = ["atlanta_emb.txt", "austin_emb.txt"]

for city in city_files: 
    print(city)
    emb = np.loadtxt(join(file_root, city))
    print('loaded')
    emb = emb[0:100000, 0:50]
    np.savetxt(city.split('_')[0] + '50_emb.txt', emb)