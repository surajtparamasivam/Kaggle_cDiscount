import io
from skimage.data import imread
from pymongo import MongoClient
import time
import bson
import json

client = MongoClient() #Makes it "good enough" for our multi-threaded use case.
train = client.cDiscount['train']
test = client.cDiscount['test']

data=train.find_one({'_id':6000})
img = imread(io.BytesIO(data['imgs'][0]['picture']))
label = data['category_id']