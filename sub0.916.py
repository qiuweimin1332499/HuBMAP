mod_path = 'C:\\Users\lenovo\Downloads\HuBMAP\hubmap-tf-with-efficientunet-256-train/'
import yaml
import pprint

with open(mod_path + 'params.yaml') as file:
    P = yaml.load(file, Loader=yaml.FullLoader)
    pprint.pprint(P)

THRESHOLD = 0.4  # preds > THRESHOLD
WINDOW = 1024
MIN_OVERLAP = 150
NEW_SIZE = P['DIM']

SUBMISSION_MODE =  'FULL' #'PUBLIC_TFREC' #
# 'PUBLIC_TFREC' = use created tfrecords for public test set with MIN_OVERLAP = 300 tiling 1024-512, ignore other (private test) data
# 'FULL' do not use tfrecords, just full submission

CHECKSUM = False  # compute mask sum for each image

import json

with open(mod_path + 'metrics.json') as json_file:
    M = json.load(json_file)
print('Model run datetime: '+M['datetime'])
print('OOF val_dice_coe: ' + str(M['oof_dice_coe']))

#! pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index -q
#! pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index -q
import numpy as np
import pandas as pd
import os
import glob
import gc

import rasterio
from rasterio.windows import Window

import pathlib
from tqdm.notebook import tqdm
import cv2

import tensorflow as tf
import efficientnet as efn
import efficientnet.tfkeras

def rle_encode_less_memory(img):
    pixels = img.T.flatten()  #转置后按列展开即是不转置的按行展开
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2  #+2？？？？？
    #np.where()[0] 表示取其行索引，np.where()[1]表示取其列索引，对于一维数组即其编号
    runs[1::2] -= runs[::2]  #后一个减前一个得到期间不会变化的长度
    return ' '.join(str(x) for x in runs)


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1  #向上取整
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window  #最后一个是从后往前取值
    x2 = (x1 + window).clip(0, x)   #clip()限制取值范围
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

identity = rasterio.Affine(1, 0, 0, 0, 1, 0) #仿射变换
fold_models = []
for fold_model_path in glob.glob(mod_path+'*.h5'):
    fold_models.append(tf.keras.models.load_model(fold_model_path,compile = False))
print(len(fold_models))

AUTO = tf.data.experimental.AUTOTUNE
image_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'x1': tf.io.FixedLenFeature([], tf.int64),
    'y1': tf.io.FixedLenFeature([], tf.int64)
}
def _parse_image(example_proto):
    example = tf.io.parse_single_example(example_proto, image_feature)
    image = tf.reshape( tf.io.decode_raw(example['image'],out_type=np.dtype('uint8')), (P['DIM'],P['DIM'], 3))
    return image, example['x1'], example['y1']

def load_dataset(filenames, ordered=True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image)
    return dataset

def get_dataset(FILENAME):
    dataset = load_dataset(FILENAME)
    dataset  = dataset.batch(64)
    dataset = dataset.prefetch(AUTO)
    return dataset


p = pathlib.Path('E:\hubmap-kidney-segmentation')
subm = {}

for i, filename in tqdm(enumerate(p.glob('test/*.tiff')), total=len(list(p.glob('test/*.tiff')))):
    #使用glob方法遍历当前目录所有.tiff文件

    print(f'{i + 1} Predicting {filename.stem}')
    #使用filename.stem获取最后的路径除去文件名
    dataset = rasterio.open(filename.as_posix(), transform=identity)
    #使用.as_posix()获取文件路径并使用rasterio.open()函数打开文件
    preds = np.zeros(dataset.shape, dtype=np.uint8)
    #用于存储一个tfrecord的预测结果
    if SUBMISSION_MODE == 'PUBLIC_TFREC' : #and MIN_OVERLAP == 300 and WINDOW == 1024 and NEW_SIZE == 256:
        #使用hubmap-tfrecords-1024-512-test文件夹的PUBLIC_TFREC的tfrecords作为输入
        print('SUBMISSION_MODE: SUBMISSION_MODE')
        fnames = glob.glob('C:\\Users\lenovo\Downloads\HuBMAP\HuBMAP_tfrecord\\test/' + filename.stem + '*.tfrec')
        #遍历hubmap-tfrecords-1024-512-test文件夹下对应的那个.tiff文件的tfrecords
        if len(fnames) > 0:  # PUBLIC TEST SET
            for FILENAME in fnames:
                pred = None
                for fold_model in fold_models:
                    tmp = fold_model.predict(get_dataset(FILENAME)).detach().cpu().numpy() / len(fold_models)
                    if pred is None:
                        pred = tmp
                    else:
                        pred += tmp   #与上面配合求平均值
                    del tmp
                    gc.collect()

                pred = tf.cast((tf.image.resize(pred, (WINDOW, WINDOW)) > THRESHOLD), tf.bool).numpy().squeeze()

                idx = 0
                for img, X1, Y1 in get_dataset(FILENAME):
                    #一个FILENAME(tfrecord)被划分成了多个WINDOW(1024)大小的模块故拼起来才是最后结果
                    for fi in range(X1.shape[0]):
                        x1 = X1[fi].numpy()
                        y1 = Y1[fi].numpy()
                        preds[x1:(x1 + WINDOW), y1:(y1 + WINDOW)] += pred[idx]
                        idx += 1

        else:  # IGNORE PRIVATE TEST SET (CREATE TFRECORDS IN FUTURE)
            pass
    else:
        print('SUBMISSION_MODE: FULL')
        # 使用所有.tiff文件作为输入,不明原因错误
        slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)

        for (x1, x2, y1, y2) in slices:
            image = dataset.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))  #？？？？？？
            image = np.moveaxis(image, 0, -1)  #0轴调至最后一轴：CBH转BHC？
            image = cv2.resize(image, (NEW_SIZE, NEW_SIZE), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = np.expand_dims(image, 0)

            pred = None
            for fold_model in fold_models:
                if pred is None:
                    pred = np.squeeze(fold_model.predict(image))
                    print(np.max(pred))
                else:
                    pred += np.squeeze(fold_model.predict(image))

            pred = pred / len(fold_models)

            pred = cv2.resize(pred, (WINDOW, WINDOW))
            preds[x1:x2, y1:y2] += (pred > THRESHOLD).astype(np.uint8)

    preds = (preds > 0.5).astype(np.uint8)

    subm[i] = {'id': filename.stem, 'predicted': rle_encode_less_memory(preds)}

    if CHECKSUM:
        print('Checksum: ' + str(np.sum(preds)))   #？？？？？？

    del preds
    gc.collect()

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('pseudo_labels.csv', index=False)
submission.head() #head()根据位置返回对象的前n行
