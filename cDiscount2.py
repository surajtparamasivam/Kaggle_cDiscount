import bson
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

out_folder = '../output/train'

# Create output folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
# Create categories folders
categories = pd.read_csv('Input/category_names.csv', index_col='category_id')

for category in tqdm(categories.index):
    os.mkdir(os.path.join(out_folder, str(category)))

num_products = 7069896  # 7069896 for train and 1768182 for test

bar = tqdm(total=num_products)
with open('Iksqnput/train.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)

    for c, d in enumerate(data):
        category = d['category_id']
        _id = d['_id']
        for e, pic in enumerate(d['imgs']):
            fname = os.path.join(out_folder, str(category), '{}-{}.jpg'.format(_id, e))
            with open(fname, 'wb') as f:
                f.write(pic['picture'])

        bar.update()