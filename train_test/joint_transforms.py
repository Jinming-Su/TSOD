import numbers
import random

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, mask_b, mask_i):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, mask_b, mask_i = t(img, mask, mask_b, mask_i)
        return img, mask, mask_b, mask_i

class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        th, tw = self.size
        return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, mask_b, mask_i):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            mask_b = ImageOps.expand(mask_b, border=self.padding, fill=0)
            mask_i = ImageOps.expand(mask_i, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask, mask_b, mask_i
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), \
                   mask_b.resize((tw, th), Image.NEAREST), mask_i.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), \
               mask_b.crop((x1, y1, x1 + tw, y1 + th)), mask_i.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, mask_b, mask_i):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
                   mask_b.transpose(Image.FLIP_LEFT_RIGHT), mask_i.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, mask_b, mask_i


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, mask_b, mask_i):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), \
               mask_b.rotate(rotate_degree, Image.NEAREST), mask_i.rotate(rotate_degree, Image.NEAREST),
