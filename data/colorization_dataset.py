import os
import os.path
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset_videvo, make_dataset_hollywood
from skimage import color
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from scipy import misc


def make_dataset(opt):
    images = []
    ref_path = 'sample_videos/ref/00/0000.jpg'
    pre_path = ref_path
    images.append([os.path.join(opt.dataroot, '0001.jpg'), pre_path, ref_path, True])

    for i in range(len(os.listdir(opt.dataroot)) - 1):
        cur_path = os.path.join(opt.dataroot, '%04d.' % (i + 2) + 'jpg')
        pre_path = os.path.join(opt.dataroot, '%04d.' % (i + 1) + 'jpg')
        images.append([cur_path, pre_path, ref_path, False])
    return images


class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the nubmer of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.testData_path = make_dataset(self.opt)
        self.transform_A = get_transform(self.opt, convert=False)
        self.transform_R = get_transform(self.opt, convert=False, force_resize=True)
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        path_A, path_P, path_R, is_first_frame = self.testData_path[index]
        im_A_l, im_A_ab = self.process_img(path_A, self.transform_A, scale=True)
        if is_first_frame:
            im_P_l, im_P_ab = self.process_img(path_P, self.transform_A)
        else:
            im_P_l, im_P_ab = torch.Tensor([0]), torch.Tensor([0])
        im_R_l, im_R_ab = self.process_img(path_R, self.transform_R)
        hist_ab = im_R_ab

        im_dict = {
            'A_l': im_A_l, 'A_ab': im_A_ab,
            'P_l': im_P_l, 'P_ab': im_P_ab,
            'R_l': im_R_l, 'R_ab': im_R_ab,
            'hist_ab': hist_ab, 'is_first_frame': is_first_frame, 'A_paths': path_A
        }

        return im_dict

    def process_img(self, im_path, transform, scale=False):
        im = Image.open(im_path).convert('RGB')
        im = transform(im)
        if not scale:
            im = np.array(im)
            if im.shape[0] % 16 != 0:
                edge_size = im.shape[0] // 16 * 16
                im = misc.imresize(im, (edge_size, im.shape[1]))
            if im.shape[1] % 16 != 0:
                edge_size = im.shape[1] // 16 * 16
                im = misc.imresize(im, (im.shape[0], edge_size))
            lab = color.rgb2lab(im).astype(np.float32)
            lab_t = transforms.ToTensor()(lab)
            l_t = lab_t[[0], ...] / 50.0 - 1.0
            ab_t = lab_t[[1, 2], ...] / 110.0
            return l_t, ab_t
        else:
            if im.size[0] % 16 != 0:
                edge_size = im.size[0] // 16 * 16
                im = im.resize((edge_size, im.size[1]), Image.BICUBIC)
            if im.size[1] % 16 != 0:
                edge_size = im.size[1] // 16 * 16
                im = im.resize((im.size[1], edge_size), Image.BICUBIC)
            ims = [np.array(im)]
            for i in [2, 4]:
                w, h = im.size
                ims = [np.array(im.resize((w//i, h//i), Image.BICUBIC))] + ims
            l_ts, ab_ts = [], []
            for im in ims:
                lab = color.rgb2lab(im).astype(np.float32)
                lab_t = transforms.ToTensor()(lab)
                l_ts.append(lab_t[[0], ...] / 50.0 - 1.0)
                ab_ts.append(lab_t[[1, 2], ...] / 110.0)
            return l_ts, ab_ts

    def __scale_width(self, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if ow <= oh:
            if (ow == target_width):
                return img
            w = target_width
            h = int(target_width * oh / ow)
        else:
            if (oh == target_width):
                return img
            h = target_width
            w = int(target_width * ow / oh)
        return img.resize((w, h), method)

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.testData_path)