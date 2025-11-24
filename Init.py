#! unzip "/content/SRGAN-master.zip" -d "/content/SRGAN-master"
#! unzip "/content/DIV2K_valid_HR.zip" -d "/content/SRGAN-master/SRGAN-master/data"
#! unzip "/content/DIV2K_train_HR.zip" -d "/content/SRGAN-master/SRGAN-master/data"

#! python --video_name test_video.mp4

# cd /home/mahendra/Documents/MPS/SRGAN-master
# pwd

##pip install pytorch-ssim
##pip install torch

#pip install pytorch-ssim
##! pip install git+https://github.com/Po-Hsun-Su/pytorch-ssim

#! pip install data_utils.py
#! pip install loss.py
#! pip install model.py


# python train.py
#! python model.py
#! python test_video.py

import torch
import torch.optim as optim
from model import Generator, Discriminator

# Define the models
netG = Generator(8)
netD = Discriminator()
optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

# Now, you can delete them later
del netG
del netD
del optimizerG
del optimizerD
torch.cuda.empty_cache()

foo = torch.tensor([1,2,3])
foo = foo.to('cuda')

print("GPU memory freed.")

#import torch
