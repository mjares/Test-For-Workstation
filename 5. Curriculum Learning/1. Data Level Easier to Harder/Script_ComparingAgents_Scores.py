import pickle
import os

import HildensiaDataset

datafolder = 'Temp'
filelist = os.listdir(datafolder)

for filename in filelist:
    saveFilename = filename
    datasetFile = open(datafolder + '/' + saveFilename, 'rb')  # Creating file to read
    dataset = pickle.load(datasetFile)

    stable = 0
    for ii in range(len(dataset)):
        if dataset[ii].stable_at_goal:
            stable = stable + 1
    print(f'{saveFilename} Stable: {stable}/{len(dataset)}')
    dataset = []

