import io

import chainer
from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image
from pymongo import MongoClient
from sklearn import preprocessing
import tensorflow as tf
import os
class DatasetfromMongoDB(dataset_mixin.DatasetMixin):
    def __init__(self, db_name='cDiscount', col_name='train', dtype=np.float32):
        self._dtype = dtype
        self._label_dtype = np.int32

        client = MongoClient('localhost', 27017)
        db = client[db_name]
        self.col = db[col_name]
        self.examples = list(self.col.find({}, {'imgs': 0}))
        self.labels = self.get_labels()

    def __len__(self):
        return len(self.examples)

    def get_labels(self):
        category_ids = [e['category_id'] for e in self.examples]
        return {cid: i for i, cid in enumerate(list(set(category_ids)))}

    def get_example(self, i):
        _id = self.examples[i]['_id']
        doc = self.col.find_one({'_id': _id})

        img = doc['imgs'][0]['picture']
        img = Image.open(io.BytesIO(img))
        img = np.asarray(img, dtype=self._dtype).transpose(2, 0, 1)
        img = img / 255.

        if chainer.config.train:
            label = self.labels[doc['category_id']]
            label = np.array(label, dtype=self._label_dtype)

            return img, label
        else:
            return img, _id

class DatasetfromMongoDB1(dataset_mixin.DatasetMixin):
    def __init__(self, db_name='cDiscount', col_name='test', dtype=np.float32):
        self._dtype = dtype
        self._label_dtype = np.int32

        client = MongoClient('localhost', 27017)
        db = client[db_name]
        self.col = db[col_name]
        self.examples = list(self.col.find({}, {'imgs': 0}))


    def __len__(self):
        return len(self.examples)



    def get_example(self, i):
        _id = self.examples[i]['_id']
        doc = self.col.find_one({'_id': _id})

        img = doc['imgs'][0]['picture']
        img = Image.open(io.BytesIO(img))
        img = np.asarray(img, dtype=self._dtype).transpose(2, 0, 1)
        img = img / 255.



Dataset_train=DatasetfromMongoDB()

for i in range(Dataset_train.__len__()):
    img, label1 = Dataset_train.get_example(i )

Dataset_test=DatasetfromMongoDB1()

for j in range(Dataset_test.__len__()):
    img_test,_=Dataset_test.get_example(i)

le = preprocessing.LabelEncoder()
le.fit(label1)
Y_ = le.transform(label1)
print(len(le.classes_))



fc = [tf.feature_column.numeric_column("x", shape=[97200])]
classifier = tf.estimator.DNNClassifier(feature_columns=fc, hidden_units=[1024, 512, 1024], n_classes=36)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(img)}, y=np.array(Y_), shuffle=True)

classifier.train(input_fn=train_input_fn, steps=10000)

test_input_fn=tf.estimator.inputs.numpy_input_fn(x={"x":np.array(img_test)},shuffle=True)

pred=classifier.predict(input_fn=test_input_fn)
os.write("results.txt",pred)





