# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set
from detectron2.utils.file_io import PathManager
import torch
import json
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.data.datasets import register_coco_panoptic
from detectron2.engine import default_argument_parser, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results
from detectron2.data.datasets.coco_panoptic import load_coco_panoptic_json
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import EventStorage
# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from torch import dist, nn
class DualBackbone(nn.Module):
    def __init__(self, backbone1, backbone2):
        super(DualBackbone, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2

    def forward(self, input1, input2):
        features1 = self.backbone1(input1)
        features2 = self.backbone2(input2)
        return features1, features2


# 全局变量来存储调用次数
call_count = 0

def count_calls_to_file(func):
    """装饰器，用于计数函数调用次数并写入文件"""
    def wrapper(*args, **kwargs):
        global call_count
        
        # 增加调用计数
        call_count += 1
        
        # 写入新的调用次数
        with open("call_count.txt", "a") as f:
            f.write(f"{str(call_count)}\n{func.__name__}\n")
            

        print(f"{func.__name__} 被调用了 {call_count} 次")
        return func(*args, **kwargs)  # 调用原始函数

    return wrapper

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                # print("cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON")
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            # print("cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON")
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            # print("cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON")
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
@count_calls_to_file
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # 设置训练数据集名称
    cfg.DATASETS.TRAIN = ("muses_panoptic_train",)
    cfg.DATASETS.TEST = ("muses_panoptic_val",)

    cfg.DATALOADER.NUM_WORKERS=1
    
    # 设置批量大小为2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR=0.0002
    # 设置最大迭代次数为10000
    cfg.SOLVER.MAX_ITER = 2000
    # cfg.SOLVER.STEPS = (200,500)
    cfg.SOLVER.AMP.ENABLED = True  # 启用混合精度训练

    # 设置帧相机网络相关配置
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False  # 禁用语义分割测试
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False  # 禁用实例分割测试
    # cfg.MODEL.NUM_CLASSES= 19
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST=0.5
    
    # 其他可能需要的帧相机网络相关配置
    # cfg.MODEL.FPN...
    # cfg.MODEL.ROI_HEADS...

    
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

def load_coco_panoptic_json_with_string_ids(json_file, image_dir, gt_dir, add_dir,meta):
    with open(json_file) as f:
        dataset = json.load(f)

    # 创建一个映射，将字符串ID映射到整数ID
    id_map = {img['file_name'].split('.')[0]: idx for idx, img in enumerate(dataset['images'])}

    # 从数据集中提取类别信息
    categories = dataset['categories']
    thing_classes = [cat['name'] for cat in categories if cat['isthing'] == 1]
    stuff_classes = [cat['name'] for cat in categories if cat['isthing'] == 0]
    thing_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories) if cat['isthing'] == 1}
    stuff_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories) if cat['isthing'] == 0}

    meta.update({
        "thing_classes": thing_classes,
        "stuff_classes": stuff_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id
    })

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][segment_info["category_id"]]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][segment_info["category_id"]]
            segment_info["isthing"] = False
        return segment_info

    ret = []
    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        if isinstance(image_id, str):
            image_id = id_map.get(image_id, len(id_map))
            id_map[image_id] = image_id
        else:
            image_id = int(image_id)

        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        
        # 添加 add_dir 的内容
        add_file_name = os.path.join(add_dir, os.path.splitext(ann['file_name'])[0] + ".jpg")  # 假设 add_dir 中有相应的 .txt 文件
        # print(os.path.exists(add_file_name),add_file_name)

        # 检查文件是否存在，并根据需要调整文件名或路径
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Panoptic segmentation file not found: {label_file}")

        # 获取图像信息，如果缺少宽度或高度，尝试从文件中获取或设置默认值
        image_info = next((img for img in dataset["images"] if img["id"] == ann["image_id"]), None)
        
        if image_info is None:
            raise ValueError(f"No image info found for image ID: {ann['image_id']}")

        width = image_info.get("width")
        height = image_info.get("height")
        
        if width is None or height is None:
            from PIL import Image
            with Image.open(image_file) as img:
                width, height = img.size

            image_info["width"] = width
            image_info["height"] = height

        segments_info = [_convert_category_id(x, meta) for x in ann.get("segments_info", [])]
        # 打开文件
        with open("load_segments_info.txt", "a") as file:
         # 遍历 segments_info 并格式化输出到文件中
            for segment in segments_info:
                category_id = segment.get("category_id")
                segment_id = segment.get("id")
                iscrowd = segment.get("iscrowd")
                isthing = segment.get("isthing")
                file.write(f"Segment ID: {segment_id}, Category ID: {category_id}, Is Crowd: {iscrowd}, Is Thing: {isthing}\n")

            ret.append(
            {
                "file_name": image_file,
                'add_file_name': add_file_name,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "width": width,
                "height": height,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    
    return ret


@count_calls_to_file
def register_my_datasets():
    # Configure the absolute paths for the datasets
    train_img_dir = "/root/autodl-tmp/data/muses/frame_camera_jpg_rename"
    lidar_dir ="/root/autodl-tmp/data/muses/projected_to_rgb/lidar_rename" 
    train_panoptic_dir = "/root/autodl-tmp/data/muses/gt_panoptic"
    train_panoptic_ann_file = "/root/autodl-tmp/data/muses/gt_panoptic/train.json"

    val_img_dir = "/root/autodl-tmp/data/muses/frame_camera_jpg_rename"
    val_panoptic_dir = "/root/autodl-tmp/data/muses/gt_panoptic"
    val_panoptic_ann_file = "/root/autodl-tmp/data/muses/gt_panoptic/val.json"

    for d in ["train", "val"]:
        name = f"muses_panoptic_{d}"
        image_dir = train_img_dir if d == "train" else val_img_dir
        panoptic_dir = train_panoptic_dir if d == "train" else val_panoptic_dir
        panoptic_json = train_panoptic_ann_file if d == "train" else val_panoptic_ann_file
        
        # if d =="train":
        #     # Register the dataset with all necessary arguments
        #     DatasetCatalog.register(
        #         name,
        #         lambda x=panoptic_json, y=image_dir, z=panoptic_dir,r=train_lidar_dir: load_coco_panoptic_json_with_string_ids(x, y, z, r, {})
        #     )
        # else:
        #     # Register the dataset with all necessary arguments
        #     DatasetCatalog.register(
        #         name,
        #         lambda x=panoptic_json, y=image_dir, z=panoptic_dir: load_coco_panoptic_json_with_string_ids_only(x, y, z, {})
        #     )

        # Register the dataset with all necessary arguments
                # Unregister the dataset if it already exists
        DatasetCatalog.register(
            name,
            lambda x=panoptic_json, y=image_dir, z=panoptic_dir,r=lidar_dir: load_coco_panoptic_json_with_string_ids(x, y, z,r,{})
        )

        # Set metadata
        metadata = MetadataCatalog.get(name)
        metadata.set(
            panoptic_root=panoptic_dir,
            image_root=image_dir,
            train_lidar_dir=lidar_dir,
            panoptic_json=panoptic_json,
            evaluator_type="coco_panoptic_seg",  # 设置评估类型为全景分割
            ignore_label=255,  # 设置忽略标签值（根据需要调整）
        )
        
        # Load and set category information
        with open(panoptic_json) as f:
            dataset = json.load(f)
        
        categories = dataset['categories']
        metadata.thing_classes = [cat['name'] for cat in categories if cat['isthing'] == 1]
        metadata.stuff_classes = [cat['name'] for cat in categories if cat['isthing'] == 0]
        
        # Create mappings from dataset category IDs to contiguous IDs
        thing_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories) if cat['isthing'] == 1}
        stuff_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories) if cat['isthing'] == 0}
         # 将类别映射保存到文件
        with open("category_mappings.txt", "w") as f:
            f.write("Thing Dataset ID to Contiguous ID:\n")
            for key, value in thing_dataset_id_to_contiguous_id.items():
                f.write(f"{key}: {value}\n")
        
            f.write("\nStuff Dataset ID to Contiguous ID:\n")
            for key, value in stuff_dataset_id_to_contiguous_id.items():
                f.write(f"{key}: {value}\n")
 
        metadata.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
        metadata.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id

def main(args):
    register_my_datasets()

    cfg = setup(args)
    # 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    writer.close()  # 关闭 TensorBoard Writer
    return trainer.train()

if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )