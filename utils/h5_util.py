import numpy as np
import h5py
from progressbar import ProgressBar
from PIL import Image

"""
This model is used to convert data files to h5 database to facilitate training
"""


# IMG_MEAN for Pascal dataset
IMG_MEAN = np.array(
    (122.67891434, 116.66876762, 104.00698793), dtype=np.float32)  # RGB


def read_images(data_list):
    with open(data_list, 'r') as f:
        data = [line.strip("\n").split(' ') for line in f]
    return data


def process_image(image, shape, resize_mode=Image.BILINEAR):
    img = Image.open(image)
    img = img.resize(shape, resize_mode)
    img.load()
    return np.asarray(img, dtype="float32")


def build_h5_voc_dataset(image_dir, label_dir, list_path, out_dir, shape, name, norm=False):
    image_list = read_images(list_path)
    images_size = len(image_list)
    print(images_size)
    dataset = h5py.File(out_dir + name + '.h5', 'w')
    dataset.create_dataset('X', (images_size, *shape, 3), dtype='f')
    dataset.create_dataset('Y', (images_size, *shape), dtype='f')
    pbar = ProgressBar()
    for index, name in pbar(enumerate(image_list)):
        image = process_image(image_dir + name[0] + '.jpg', shape)
        label = process_image(label_dir + name[0] + '.png', shape, Image.NEAREST)
        image -= IMG_MEAN
        image = image / 255. if norm else image
        dataset['X'][index], dataset['Y'][index] = image, label
    dataset.close()

if __name__ == '__main__':
    shape = (256, 256)
    image_dir = '../dataset/VOCdevkit/VOC2012/JPEGImages/'
    label_dir = '../dataset/VOCdevkit/VOC2012/SegmentationClass/'
    list_dir = '../dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/'
    output_dir = '../dataset/VOCdevkit/VOC2012/'

    data_files = {
        'training': 'train.txt',
        'validation': 'trainval.txt',
        'testing': 'val.txt'
    }
    for name, list_path in data_files.items():
        build_h5_voc_dataset(image_dir, label_dir, list_dir + list_path, output_dir, shape, name)