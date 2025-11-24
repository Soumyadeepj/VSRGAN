
1)loss.py
Vgg16---- earlier used
Vgg19-----Feature map for calculating loss function 

2) data_utils.py

def train_lr_transform(crop_size, upscale_factor,sigma =3.0):        
# Resize the image (downsample)
#Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), earlier used
Resize(crop_size // upscale_factor, interpolation=Image.LANCZOS), using now




