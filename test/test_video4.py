import argparse
import cv2
import numpy as np
import torch
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0)
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from model import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--video_name', default='test_videoLion_360p_clip.mp4', type=str, help='test low resolution video name')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu') ))

    videoCapture = cv2.VideoCapture(VIDEO_NAME)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    sr_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
                     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
    #output_sr_name = 'out_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    output_sr_name = '/kaggle/working/out_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    sr_video_writer = cv2.VideoWriter(output_sr_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, sr_video_size)

    success, frame = videoCapture.read()
    test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')
    for index in test_bar:
        if success:
            # Convert frame to tensor
            image = ToTensor()(frame).unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()

            # Disable gradient tracking for inference
            with torch.no_grad():
                # Forward pass through the model
                out = model(image)
                out_img = out[0].cpu().detach().numpy()  # Convert output to numpy for saving
                out_img *= 255.0
                out_img = (np.uint8(out_img)).transpose((1, 2, 0))

            # Write the super-resolved frame to video
            sr_video_writer.write(out_img)

            # Read the next frame
            success, frame = videoCapture.read()
    
    videoCapture.release()
    sr_video_writer.release()

    print(f"Super-resolved video saved as {output_sr_name}")
