import json
import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog

which_folder = 1

if which_folder == 0:
    savedir_base_json = "~/dev/ril-digitaltwin/scripts/"
    savedir_base_images = "~/dev/ril-digitaltwin/scripts/imgs/512/"
elif which_folder == 1:
    # savedir_base_json = "/mnt/home/projects/digitaltwin/data/generatorv7-small"
    savedir_base_json = "/mnt/home/projects/digitaltwin/data2/gen7panoptic/gen7"
    savedir_base_images = savedir_base_json

PATH_IMAGES = os.path.expanduser(f"{savedir_base_images}/generatorv7")
PATH_PANOPT = os.path.expanduser(f"{savedir_base_images}/generatorv7_panoptic")
DATA_JSON = os.path.join(PATH_PANOPT, "00000_dsinfo.json")

DATASET_NAME = "rilv7"
DATASET_NAME_TEST = "rilv7-test"
TEST_SPLIT = 0.1  # 10 %

data_json = json.load(open(DATA_JSON, "r"))
categories = data_json["categories"]

len_data = len(data_json["annotations"])
len_test = int(len_data * TEST_SPLIT)
len_train = len_data - len_test

data_json_train = data_json.copy()
data_json_train["annotations"] = data_json_train["annotations"][:len_train]
data_json_train_path = os.path.join(PATH_PANOPT, "00000_dsinfo_train.json")
json.dump(data_json_train, open(data_json_train_path, "w"))

data_json_test = data_json.copy()
data_json_test["annotations"] = data_json_test["annotations"][len_train:]
data_json_test_path = os.path.join(PATH_PANOPT, "00000_dsinfo_test.json")
json.dump(data_json_test, open(data_json_test_path, "w"))


def convert_category_id(segment_info, meta):
    if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
        segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        segment_info["isthing"] = True
    else:
        segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        segment_info["isthing"] = False
    return segment_info


def adjust_meta_for_vis(segment_info, meta):
    if (
        segment_info["category_id"]
        in meta["thing_dataset_id_to_contiguous_id"].values()
    ):
        segment_info["category_id"] += 1
    return segment_info


def replace_paths(info, path_inputs, path_panoptic, metadata, start, end):
    out = []
    for x in info["annotations"][start:end]:
        x["pan_seg_file_name"] = f"{path_panoptic}/{x['pan_seg_file_name']}"
        if "sem_seg_file_name" in x:
            del x["sem_seg_file_name"]
        # x["sem_seg_file_name"] = f"{path_semseg}/{x['file_name']}"  # FIXME
        x["file_name"] = f"{path_inputs}/{x['file_name']}"
        x["segments_info"] = [
            convert_category_id(y, metadata) for y in x["segments_info"]
        ]
        out.append(x)
    return out


# TODO integrate this and basically redo this file from scratch with new knawledge
def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in categories]
    # thing_classes = [k["name"] for k in categories if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["stuff_classes"] = stuff_classes

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


metadata = get_metadata()
data_train = replace_paths(data_json, PATH_IMAGES, PATH_PANOPT, metadata, 0, len_train)
data_test = replace_paths(
    data_json, PATH_IMAGES, PATH_PANOPT, metadata, len_train, len_data
)


def get_data_train():  # this is stupid -.-'
    return data_train


def get_data_test():  # this is stupid -.-'
    return data_test


DatasetCatalog.register(DATASET_NAME, get_data_train)
DatasetCatalog.register(DATASET_NAME_TEST, get_data_test)

full_metadata = {
    "panoptic_root": PATH_PANOPT,
    "image_root": PATH_IMAGES,
    "evaluator_type": "ril_panoptic",
    "ignore_label": 1,
    "label_divisor": 1000,
    # "panoptic_json": DATA_JSON,
}
full_metadata.update(metadata)


MetadataCatalog.get(DATASET_NAME).set(
    panoptic_json=data_json_train_path, **full_metadata
)
MetadataCatalog.get(DATASET_NAME_TEST).set(
    panoptic_json=data_json_test_path, **full_metadata
)


# TODO then add this to toolkit
# todo install mask2former on toolkit
# TODO see if we can call the yaml file with the train_net script

# python train_net.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001

if __name__ == "__main__":
    from pprint import pprint

    data = get_data_train()
    print(len(data))
    print(data[2])
    import random
    from detectron2.utils.visualizer import Visualizer
    import cv2

    meta = MetadataCatalog.get(DATASET_NAME)

    for d in random.sample(data, 3):
        # d["segments_info"] = [adjust_meta_for_vis(x, metadata) for x in d["segments_info"]]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(f"img {d['file_name']}", vis.get_image()[:, :, ::-1])
        cv2.waitKey(-1)
