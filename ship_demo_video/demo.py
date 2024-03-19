# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from predictor import VisualizationDemo

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
# constants
WINDOW_NAME = "mask2former video demo"



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        "this will be treated as frames of a video",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    if no_args := True: 
        args.config_file = "/home/viewegm/video_segmentation/video_segmentation/ship_demo_video/configs/ship_R50.yaml"
        args.output = "/home/viewegm/visualizations"
        args.input = ['/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00001.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00002.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00003.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00004.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00005.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00006.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00007.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00008.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00009.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00010.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00011.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00012.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00013.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00014.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00015.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00016.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00017.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00018.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00019.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00020.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00021.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00022.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00023.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00024.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00025.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00026.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00027.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00028.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00029.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00030.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00031.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00032.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00033.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00034.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00035.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00036.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00037.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00038.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00039.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00040.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00041.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00042.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00043.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00044.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00045.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00046.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00047.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00048.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00049.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00050.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00051.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00052.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00053.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00054.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00055.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00056.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00057.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00058.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00059.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00060.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00061.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00062.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00063.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00064.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00065.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00066.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00067.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00068.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00069.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00070.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00071.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00072.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00073.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00074.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00075.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00076.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00077.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00078.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00079.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00080.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00081.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00082.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00083.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00084.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00085.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00086.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00087.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00088.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00089.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00090.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00091.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00092.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00093.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00094.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00095.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00096.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00097.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00098.png', '/home/viewegm/data/ytvis_2021/train/JPEGImages/Images1/Image00099.png']
        args.opts = ["MODEL.WEIGHTS", "/home/viewegm/models/mask2former_video/trained_models/model_0000099.pth"]
        args.confidence_threshold = 0.5


    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        vid_frames = []
        for path in args.input:
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        logger.info(
            "detected {} instances per frame in {:.2f}s".format(
                len(predictions["pred_scores"]), time.time() - start_time
            )
        )

        if args.output:
            if args.save_frames:
                for path, _vis_output in zip(args.input, visualized_output):
                    out_filename = os.path.join(args.   output, os.path.basename(path))
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        
        vid_frames = []
        while video.isOpened():
            success, frame = video.read()
            if success:
                vid_frames.append(frame)
            else:
                break

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        logger.info(
            "detected {} instances per frame in {:.2f}s".format(
                len(predictions["pred_scores"]), time.time() - start_time
            )
        )

        if args.output:
            if args.save_frames:
                for idx, _vis_output in enumerate(visualized_output):
                    out_filename = os.path.join(args.output, f"{idx}.jpg")
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()
