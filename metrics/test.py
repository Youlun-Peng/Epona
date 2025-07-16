import os
import time
import cv2
import torch
from calculate_fvd import calculate_fvd
from PIL import Image
import numpy as np
import tqdm
import math
import torch.nn.functional as F
from torchvision.utils import save_image
import re

def find_latest_checkpoint(save_path, test_folder):
    pattern = re.compile(rf"{test_folder}-(\d+)\.pt")
    max_index = -1
    latest_checkpoint = None
    
    if not os.path.exists(save_path):
        print(f"Directory {save_path} does not exist.")
        return None, 0
    
    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                latest_checkpoint = filename
    
    if latest_checkpoint:
        checkpoint_path = os.path.join(save_path, latest_checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.shape)
        return checkpoint, max_index
    
    return None, 0

# ps: pixel value should be in [0, 1]!

# NUMBER_OF_VIDEOS = 8
# VIDEO_LENGTH = 30
# CHANNEL = 3
# HEIGHT = 64
# WIDTH = 128
# videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, HEIGHT, WIDTH, requires_grad=False)
# videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, HEIGHT, WIDTH, requires_grad=False)
device = torch.device("cuda")
start_id = 0
end_id = 1628
img_start = 10
img_end = 50
gt_folder = "val-gt"
test_folder = "val-938k-gt-50-cond2"
save_path = "metrics"
os.makedirs(save_path, exist_ok=True)


gtpath = f"{save_path}/{gt_folder}.pt"
if os.path.exists(gtpath):
    videos1 = torch.load(gtpath)
    print("Load videos1", videos1.shape)
else:
    frames = []
    # device = torch.device("cpu")
    video_tmp = []
    # for index in tqdm.tqdm(range(100)): # 2692, 5384)): #5384 2692
    for index in tqdm.tqdm(range(start_id, end_id)): # 5384
    # for index in tqdm.tqdm(range(2692)): # 5384
        # step = index // 8
        # rank = index % 8
        step = index
        rank = 0
        for i in range(img_start, img_end):
            gfile = Image.open(f"test_videos/{gt_folder}/sliding_{step}/{i}.png")
            frame = gfile.convert('RGB')
            frame_tensor = torch.tensor(np.array(gfile.convert('RGB'))/255.).permute(2,0,1).to(torch.float32)
            
            # video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
            video_single = frame_tensor.unsqueeze(0)
            t, c, h, w = video_single.shape
            resolution = 224
            # scale shorter side to resolution
            scale = resolution / min(h, w)
            if h < w:
                target_size = (resolution, math.ceil(w * scale))
            else:
                target_size = (math.ceil(h * scale), resolution)
            video_single = F.interpolate(video_single, size=target_size, mode='bilinear',
                                align_corners=False)

            # center crop
            t, c, h, w = video_single.shape
            w_start = (w - resolution) // 2
            h_start = (h - resolution) // 2
            video_single = video_single[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
            # video_single = video_single.permute(1, 0, 2, 3).contiguous() # CTHW

            # video_single -= 0.5
            frame_tensor = video_single[0]

            frames.append(frame_tensor)
            # save_image(frame_tensor, f"recon_{index}_{i}.png")
        video_tmp.append(torch.stack(frames))
        frames = []
    videos1 = torch.stack(video_tmp)
    print("Get videos1", videos1.shape)
    torch.save(videos1, gtpath)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载损坏的图片

frames = []
video_tmp = []
# for index in tqdm.tqdm(range(2842)): # 2692, 5384)): # 5384
checkpoint, start_id = find_latest_checkpoint(save_path, test_folder)
print(f"Starting from index: {start_id}")

for index in tqdm.tqdm(range(start_id, end_id)): # 5384
    step = index
    rank = 0
    for i in range(img_start, img_end):
        img_path = f"test_videos/{test_folder}/sliding_{step}/{i}.png"
        retry_count = 5
        for attempt in range(retry_count):
            try:
                gfile = Image.open(img_path)
                frame_tensor = torch.tensor(np.array(gfile.convert('RGB'))/255.).permute(2,0,1).to(torch.float32)

                # img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 读取为 BGR 格式
                # if img is None:
                #     raise ValueError("cv2 读取失败，可能是损坏文件")
                # frame_tensor = torch.tensor(img[:, :, ::-1] / 255.).permute(2, 0, 1).to(torch.float32)
            except:
                if attempt < retry_count - 1:
                    print(f"读取失败，重试 {attempt+1}/{retry_count}：{img_path}")
                    time.sleep(0.5)  # 等待 0.5 秒后重试
                else:
                    print(f"最终失败，跳过：{img_path}")
                    videos2 = torch.stack(video_tmp)
                    if checkpoint is not None:
                        videos2 = torch.cat([checkpoint, videos2], dim=0)
                    print("Current videos2", videos2.shape)
                    torch.save(videos2, f"{save_path}/{test_folder}-{index}.pt")
                    exit()

        video_single = frame_tensor.unsqueeze(0)
        t, c, h, w = video_single.shape
        resolution = 224
        # scale shorter side to resolution
        scale = resolution / min(h, w)
        if h < w:
            target_size = (resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), resolution)
        video_single = F.interpolate(video_single, size=target_size, mode='bilinear',
                            align_corners=False)
        # print("?video_single", video_single.shape)

        # center crop
        t, c, h, w = video_single.shape
        w_start = (w - resolution) // 2
        h_start = (h - resolution) // 2
        video_single = video_single[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
        # video_single = video_single.permute(1, 0, 2, 3).contiguous() # CTHW

        # video_single -= 0.5
        frame_tensor = video_single[0]
        frames.append(frame_tensor)
        # save_image(frame_tensor, f"gt_{index}_{i}.png")
    video_tmp.append(torch.stack(frames))
    frames = []
videos2 = torch.stack(video_tmp)
if checkpoint is not None:
    videos2 = torch.cat([checkpoint, videos2], dim=0)
print("Get videos2", videos2.shape)
torch.save(videos2, f"{save_path}/{test_folder}.pt")


import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
print(json.dumps(result, indent=4))
