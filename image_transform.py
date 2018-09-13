from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

class ImageTransform:
    """
    Use imgaug library to do image augmentation.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
