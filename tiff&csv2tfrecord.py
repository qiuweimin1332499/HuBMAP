import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm.notebook import tqdm
import tensorflow as tf
import gc
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

orig = 1024
sz = 256 #128 #256 #the size of tiles
reduce = orig//sz  #reduce the original images by 'reduce' times
MASKS = 'E:/hubmap-kidney-segmentation/train.csv'
DATA = 'E:/hubmap-kidney-segmentation/train/'
s_th = 40  #saturation blancking threshold
p_th = 1000*(sz//256)**2 #threshold for the minimum number of pixels

#top_n = 8 # or only first 5 tiff files for train, train2 and test will be processed due to output 20gb limit

#functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

df_masks = pd.read_csv(MASKS).set_index('id')
df_masks.head()

### Thank you @iafoss
class HuBMAPDataset(Dataset):  #以pytorch中的dataset类为父类，构造此类有利于逐步提取数据降低对内存的要求。
    def __init__(self, idx, sz=sz, reduce=reduce, encs=None):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), num_threads='all_cpus')
        # some images have issues with format
        # and must be saved correctly before reading with rasterio
        if self.data.count == 1:
            tiff.imwrite('tmp.tiff', tiff.imread(os.path.join(DATA, idx + '.tiff'))) # , photometric='rgb'
            self.data = rasterio.open('tmp.tiff', num_threads='all_cpus')
            gc.collect()
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz + shift
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz + shift
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz
        self.mask = enc2mask(encs, (self.shape[1], self.shape[0])) if encs is not None else None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding (like in the previous version of the kernel)
        # then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz

        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask = np.zeros((self.sz, self.sz), np.uint8)
        # mapping the loade region to the tile

        img[(p00-x0):(p01-x0), (p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                 window=Window.from_slices( (p00, p01), (p10, p11))), 0, -1)
        if self.mask is not None: mask[(p00-x0):(p01-x0), (p10-y0):(p11-y0)] = self.mask[p00:p01, p10:p11]

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz//reduce, self.sz//reduce),
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.sz//reduce, self.sz//reduce),
                              interpolation=cv2.INTER_NEAREST)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # return -1 for empty images
        return img, mask, (-1 if (s>s_th).sum()<=p_th or img.sum()<=p_th else idx)


# The following function can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, mask):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'image': _bytes_feature(image),
        'mask': _bytes_feature(mask),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


OUT_TRAIN = 'train'
if(not os.path.exists(f'{OUT_TRAIN}')):
    os.makedirs(f'{OUT_TRAIN}')
x_tot, x2_tot = [], []
shift = 0
msks_strat = []
for index, encs in tqdm(df_masks.iterrows(), disable=True):
    print(index)
    # read image and generate the mask
    ds = HuBMAPDataset(index, encs=encs)
    filename = 'train/' + index + '.tfrec'
    cnt = 0
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(ds)):
            im, m, idx = ds[i]
            if idx < 0: continue
            x_tot.append((im / 255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((im / 255.0) ** 2).reshape(-1, 3).mean(0))
            # write data
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            example = serialize_example(im.tobytes(), m.tobytes())
            writer.write(example)
            cnt += 1
    os.rename(filename, 'train/' + index + '-' + str(cnt) + '.tfrec')
    msks_strat.append(cnt)
    gc.collect()
# image stats
img_avr = np.array(x_tot).mean(0)
img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
print('mean:', img_avr, ', std:', img_std)




import re
import glob
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
train_images = glob.glob('train/*.tfrec')
ctraini = count_data_items(train_images)
print(f'Num train images: {ctraini}')

DIM = sz
mini_size = 64
N = 8

def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(tf.io.decode_raw(single_example['image'], out_type=np.dtype('uint8')), (DIM, DIM, 3))
    mask = tf.reshape(tf.io.decode_raw(single_example['mask'], out_type='bool'), (DIM, DIM, 1))

    image = tf.image.resize(image, (mini_size, mini_size)) / 255.0
    mask = tf.image.resize(tf.cast(mask, 'uint8'), (mini_size, mini_size))
    return image, mask

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda ex: _parse_image_function(ex))
    return dataset

def get_dataset(FILENAME):
    dataset = load_dataset(FILENAME)
    dataset = dataset.batch(N * N)
    return dataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.segmentation import mark_boundaries
for imgs, masks in get_dataset(train_images[0]).take(1):
    pass
plt.figure(figsize=(N, N))
gs1 = gridspec.GridSpec(N, N)
for i in range(N * N):
    # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.imshow(mark_boundaries(imgs[i], masks[i].numpy().squeeze().astype('bool')))
#plt.show()
plt.ion()
plt.pause(4)
plt.close()




OUT_TRAIN = 'train2'
if(not os.path.exists(f'{OUT_TRAIN}')):
    os.makedirs(f'{OUT_TRAIN}')
x_tot, x2_tot = [], []
shift = orig//2
for index, encs in tqdm(df_masks.iterrows(), disable=True):
    print(index)
    # read image and generate the mask
    ds = HuBMAPDataset(index, encs=encs)
    filename = 'train2/' + index + '.tfrec'
    cnt = 0
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(ds)):
            im, m, idx = ds[i]
            if idx < 0: continue
            x_tot.append((im / 255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((im / 255.0) ** 2).reshape(-1, 3).mean(0))
            # write data
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            example = serialize_example(im.tobytes(), m.tobytes())
            writer.write(example)
            cnt += 1
    msks_strat.append(cnt)
    os.rename(filename, 'train2/' + index + '-' + str(cnt) + '.tfrec')
    gc.collect()
# image stats
img_avr = np.array(x_tot).mean(0)
img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
print('mean:', img_avr, ', std:', img_std)




train2_images = glob.glob('train2/*.tfrec')
ctrain2i = count_data_items(train2_images)
print(f'Num train2 images: {ctrain2i}')

for imgs, masks in get_dataset(train2_images[0]).take(1):
    pass
plt.figure(figsize = (N,N))
for i in range(N*N):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.imshow(mark_boundaries(imgs[i], masks[i].numpy().squeeze().astype('bool')))

plt.ion()
plt.pause(4)
plt.close()
#plt.show()





WINDOW = orig  # 1024
MIN_OVERLAP = 300
NEW_SIZE = sz  # 512
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

def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, x1, y1):
    feature = {
        'image': _bytes_feature(image),
        'x1': _int64_feature(x1),
        'y1': _int64_feature(y1)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


p = pathlib.Path('E:/BaiduNetdiskDownload/hubmap-kidney-segmentation')
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
os.makedirs('test', exist_ok=True)
for i, filename in tqdm(enumerate(p.glob('test/*.tiff')),
                        total=len(list(p.glob('test/*.tiff'))),
                        disable=True):

    print(f'{i + 1} Creating tfrecords for image: {filename.stem}')
    dataset = rasterio.open(filename.as_posix(), transform=identity)
    slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)

    if dataset.count != 3:
        layers = [rasterio.open(subd) for subd in dataset.subdatasets]

    print(slices.shape[0])
    cnt = 0
    part = 0
    fname = f'test/{filename.stem}-part{part}.tfrec'
    writer = tf.io.TFRecordWriter(fname)
    for (x1, x2, y1, y2) in slices:
        if cnt > 999:
            writer.close()
            os.rename(fname, f'test/{filename.stem}-part{part}-{cnt}.tfrec')
            part += 1
            fname = f'test/{filename.stem}-part{part}.tfrec'
            writer = tf.io.TFRecordWriter(fname)
            cnt = 0

        if dataset.count == 3:
            image = dataset.read([1, 2, 3],
                                 window=Window.from_slices((x1, x2), (y1, y2)))
            image = np.moveaxis(image, 0, -1)
        else:
            image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
            for fl in range(3):
                image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))

        image = cv2.resize(image, (NEW_SIZE, NEW_SIZE), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        example = serialize_example(image.tobytes(), x1, y1)
        writer.write(example)
        cnt += 1
    writer.close()
    del writer
    os.rename(fname, f'test/{filename.stem}-part{part}-{cnt}.tfrec')
    gc.collect()
test_images = glob.glob('test/*.tfrec')
ctesti = count_data_items(test_images)
print(f'Num test images: {ctesti}')




DIM = sz
mini_size = 64
N = 8

def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'x1': tf.io.FixedLenFeature([], tf.int64),
        'y1': tf.io.FixedLenFeature([], tf.int64)
    }
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape( tf.io.decode_raw(single_example['image'],out_type=np.dtype('uint8')), (DIM,DIM, 3))
    x1 = single_example['x1']
    y1 = single_example['y1']
    image = tf.image.resize(image,(mini_size,mini_size))/255.0
    return image, x1, y1

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda ex: _parse_image_function(ex))
    return dataset

def get_dataset(FILENAME):
    dataset = load_dataset(FILENAME)
    dataset = dataset.batch(N*N)
    return dataset
for imgs, x1, y1 in get_dataset(test_images[1]).take(2):
    pass
plt.figure(figsize = (N,N))
gs1 = gridspec.GridSpec(N,N)
for i in range(N*N):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.set_title(f'{x1[i]}; {y1[i]}', fontsize=6)
    ax1.imshow(imgs[i])
plt.show()