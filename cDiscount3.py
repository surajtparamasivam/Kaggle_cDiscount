import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from io import BytesIO
import cv2
import bson
from skimage.data import imread
import matplotlib.pyplot as plt

category_data=pd.read_csv("input/category_names.csv")
print("Total categories are:", len(category_data))
category_data.head(0)

def get_the_data(path):
    data = bson.decode_file_iter(open(path, 'rb'))

    images=[]
    category=[]
    prod_to_category = dict()
    images_per_category=[]
    flag=0
    for c, d in enumerate(data):
        product_id = d['_id']
        category_id = d['category_id'] # This won't be in Test data
        #prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            category.append(category_id)
            picture = imread(BytesIO(pic['picture']))
            #picture=pic['picture']
            images.append(picture)
            flag+=1
            #break
        if(flag>5000):
            break
    return category, images

product_category,image=get_the_data('input/train_example.bson')

image_gr=np.array(image)
plt.imshow(image_gr[0])
# plt.show()
print(image_gr.shape)

y,rev_labels=pd.factorize(product_category)
print(y,rev_labels)

import tensorflow as tf

fc=[tf.contrib.layers.real_valued_column("x",dimension=2)]
classifier=tf.estimator.LinearClassifier(feature_columns=fc,n_classes=36)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(image_gr)},
    y=np.array(rev_labels),
    num_epochs=None,
    shuffle=True)
classifier.train(input_fn=train_input_fn,steps=200)