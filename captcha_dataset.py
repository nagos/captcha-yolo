import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from image import ImageCaptcha
import numpy
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, size=1024, digit_out = 0):
        self.size = size
        self.img_width = 110
        self.img_height = 40
        self.digits = 3
        self.positions = 13
        self.position_width = self.img_width/self.positions
        self.digit_out = digit_out
        self.image = ImageCaptcha(width=self.img_width, height=self.img_height, font_sizes=[40, 30], fonts=[
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf",
            "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf",
            "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-BI.ttf",
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        offset = numpy.random.randint(0, 50)
        d = numpy.random.randint(0, 9+1, self.digits)
        img, b = self.image.generate_image("".join([str(x) for x in d]), offset=offset)
        convert_tensor = transforms.ToTensor()

        label = numpy.zeros((12, 1, self.positions))
        # calculate ground truth for each possible position
        for p in range(self.positions):
            self.calc(p, b, d, label)
        if self.digit_out==0:
            return convert_tensor(img), torch.tensor(label)
        else:
            return convert_tensor(img), torch.tensor(label), img, d
    
    def calc(self, p, b, d, label):
        position_center = self.position_width*p + self.position_width/2
        digit_offset = numpy.array(b) - position_center
        nearest_digit = numpy.argmin(numpy.abs(digit_offset))
        nearest_offset = digit_offset[nearest_digit]
        if -self.position_width < nearest_offset < self.position_width:
            v = 1
        else:
            v = 0
        dx = 0.5+(nearest_offset)/self.position_width/2
        dx = numpy.clip(dx, 0, 1)
            
        label[0][0][p] = v
        label[-1][0][p] = dx
        label[1+d[nearest_digit]][0][p] = 1
