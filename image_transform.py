from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import os
from config import Config
import cv2
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, 'data_process', 'fold3')

class ImageTransform:
    """
    Use imgaug library to do image augmentation.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.ElasticTransformation(0.1)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def check_cv2_imwrite(file_path, file):
    if not os.path.exists(file_path):
        cv2.imwrite(file_path, file)


def img_aug(img):
    """Do augmentation with different combination on each training batch
    """
    # without additional operations
    # according to the paper, operations such as shearing, fliping horizontal/vertical,
    # rotating, zooming and channel shifting will be apply
    random = iaa.Sequential([
        iaa.SomeOf((0, 7), [
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Affine(shear=(-16, 16)),
            iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)}),
            iaa.Affine(rotate=(-90, 90)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
            #iaa.PerspectiveTransform(scale=(0.01, 0.1))
        ])
    ])
    aug = random.augment_image(img)

    return aug


if __name__ == '__main__':
    print(DATA_DIR)
    augmentation_num = Config.aug
    for (root, dir, filenames) in os.walk(DATA_DIR):
        #print("root: {}, dir: {}, filenames: {}".format(root, dir, filenames))
        if len(filenames) != 0 and filenames[0].endswith('.png'):
            for i, img_file in enumerate(filenames):
                img_name, _ = img_file.split('.')
                print('img_name: {}, full path: {}'.format(img_name, os.path.join(root, img_file)))
                for j in range(augmentation_num):
                    cur_img = cv2.imread(os.path.join(root, img_file))
                    aug_img = img_aug(cur_img)
                    aug_img_name = img_name + 'AUG_' +str(j) + '.png'
                    check_cv2_imwrite(os.path.join(root, aug_img_name), aug_img)

