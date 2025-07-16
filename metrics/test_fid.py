import os
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from shutil import copyfile
from PIL import Image
from tqdm import tqdm
import torch
from metrics.pytorch_fid.fid_score import calculate_fid_given_paths
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载损坏的图片

def collect_images(src_root, dst_root, start, end):
    """收集指定编号范围的图片到临时文件夹，并调整大小为 256x512，以便计算FID"""
    os.makedirs(dst_root, exist_ok=True)
    for subdir in tqdm(sorted(os.listdir(src_root))):
        subdir_path = os.path.join(src_root, subdir)
        if os.path.isdir(subdir_path):
            for i in range(start, end + 1):
                img_name = f"{i}.png"
                src_img_path = os.path.join(subdir_path, img_name)
                if os.path.exists(src_img_path):
                    dst_img_path = os.path.join(dst_root, f"{subdir}_{img_name}")
                    
                    # 读取并调整图片大小
                    with Image.open(src_img_path) as img:
                        img = img.resize((512, 256), Image.BICUBIC)
                        img.save(dst_img_path)

def calculate_fid(gt_path, gen_path):
    """使用pytorch-fid计算FID"""
    # result = subprocess.run(["python", "-m", "pytorch_fid", gt_path, gen_path], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    fid_value = calculate_fid_given_paths([gt_path, gen_path], batch_size=50, device="cuda" if torch.cuda.is_available() else "cpu", dims=2048)
    print(f"FID: {fid_value}")

def main(gt_folder, gen_folder, start, end):
    with tempfile.TemporaryDirectory() as temp_dir:
        gt_temp = os.path.join(temp_dir, "gt")
        gen_temp = os.path.join(temp_dir, "generated")
        
        collect_images(gt_folder, gt_temp, start, end)
        collect_images(gen_folder, gen_temp, start, end)
        
        print("GT images:", len(os.listdir(gt_temp)))
        print("Generated images:", len(os.listdir(gen_temp)))

        calculate_fid(gt_temp, gen_temp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="Ground truth 图片的根目录")
    parser.add_argument("--gen", type=str, required=True, help="Generated 图片的根目录")
    parser.add_argument("--start", type=int, required=True, help="开始的图片编号")
    parser.add_argument("--end", type=int, required=True, help="结束的图片编号")
    args = parser.parse_args()
    
    print("Configs:", args.gt, args.gen, args.start, args.end)
    main(args.gt, args.gen, args.start, args.end)
