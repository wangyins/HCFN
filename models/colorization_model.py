from .main_model import MainModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np
import cv2


class ColorizationModel(MainModel):
    """This is a subclass of Pix2PixModel for image colorization (black & white image -> colorful images).

    The model training requires '-dataset_model colorization' dataset.
    It trains a pix2pix model, mapping from L channel to ab channels in Lab color space.
    By default, the colorization dataset will automatically set '--input_nc 1' and '--output_nc 2'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        MainModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        # reuse the pix2pix model
        MainModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A_l_0', 'real_A_rgb', 'real_R_rgb', 'fake_R_rgb']

    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def tensor2gray(self, im):
        im = im[0].data.cpu().float().numpy()
        im = np.transpose(im.astype(np.float64), (1, 2, 0))
        im = np.repeat(im, 3, axis=-1) * 255
        return im

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        r_A_l = self.real_A_l
        r_A_ab = self.real_A_ab
        f_A_ab = self.fake_imgs

        self.real_A_l_0 = r_A_l[-1]
        self.real_A_rgb = self.lab2rgb(r_A_l[-1], r_A_ab[-1])
        self.real_R_rgb = self.lab2rgb(self.real_R_l, self.real_R_ab)
        self.real_R_rgb = cv2.resize(self.real_R_rgb, (self.real_A_rgb.shape[1], self.real_A_rgb.shape[0]))
        self.fake_R_rgb = []
        for i in range(3):
            self.fake_R_rgb += [self.lab2rgb(r_A_l[i], f_A_ab[i])]
            if i != 2:
                self.fake_R_rgb[i] = cv2.resize(self.fake_R_rgb[i], (self.real_A_rgb.shape[1], self.real_A_rgb.shape[0]))
