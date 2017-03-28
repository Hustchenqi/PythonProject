"""
used for ImageNet
"""
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# read rgb and groundtruth from txt file, rgbpath and groundtruth path in each line split with ' '
def make_dataset(root, filename):
    images = []
    path = os.path.join(root, filename)
    with open(path) as fp:
        for line in fp:
            files = line.split('\t')
            # files[2] ==> RGB path, files[1] ==> class value
            abspath = os.path.join(root, files[2])
            item = (abspath[:-1], int(files[1]))
            images.append(item)
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, filename, transform = None, target_transform = None, loader = default_loader):
        imgs = make_dataset(root, filename)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in file: {}".format(filename)))
        self.root = root
        self.filename = filename
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
