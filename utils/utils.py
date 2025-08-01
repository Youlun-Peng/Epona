import os, math
import logging
import numpy as np
from datetime import datetime
import locale

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def format_number(num, use_unit=True):
    locale.setlocale(locale.LC_ALL, '')  # 设置为本地数字格式
    formatted_num = locale.format_string("%d", num, grouping=True)
    if use_unit:
        if num >= 1e9:
            return f"{num / 1e9:.2f}B ({formatted_num})"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M ({formatted_num})"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K ({formatted_num})"
    return formatted_num

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def rgb2ycbcr(img_np):
    h, w, _ = img_np.shape
    y_map = np.zeros((h, w)).astype(np.float32)
    Y = 0.257*img_np[:,:, 2]+0.504*img_np[:,:, 1]+0.098*img_np[:,:, 0]+16

    return Y

