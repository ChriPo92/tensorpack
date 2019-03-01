import numpy as np
import scipy.io as scio
import os
import tqdm
import json
import cv2

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg

class YCBVDetectionDataset:
    """
       A singleton to load datasets, evaluate results, and provide metadata.

       To use your own dataset that's not in COCO format, rewrite all methods of this class.
       """

    def __init__(self):
        """
        This function is responsible for setting the dataset-specific
        attributes in both cfg and self.
        """
        self.num_category = cfg.DATA.NUM_CATEGORY = 21
        self.num_classes = self.num_category + 1
        self.image_sets = os.path.join(cfg.DATA.BASEDIR, "image_sets/")
        self.image_dir = os.path.join(cfg.DATA.BASEDIR, "data/")
        classes_dict = self.load_id_classes_dict()
        class_names = []
        self.YCB_id_to_category_id = {}
        self.category_id_to_YCB_id = {}
        self.category_name_to_YCB_id = {}
        self.YCB_id_to_category_name = {}
        self.category_name_to_category_id = {}
        self.category_id_to_category_name = {}
        for key, value in classes_dict.items():
            self.YCB_id_to_category_id[value["ycb_id"]] = key
            self.category_id_to_YCB_id[key] = value["ycb_id"]
            self.category_name_to_YCB_id[value["name"]] = value["ycb_id"]
            self.YCB_id_to_category_name[value["ycb_id"]] = value["name"]
            self.category_id_to_category_name[key] = value["name"]
            self.category_name_to_category_id[value["name"]] = key
            class_names.append(value["name"])
        self.class_names = cfg.DATA.CLASS_NAMES = ["BG"] + class_names

    def load_id_classes_dict(self):
        id_class = {}
        with open(os.path.join(self.image_sets, "classes.txt")) as f:
            for i, line in enumerate(f):
                ycb_id = int(line[:3])
                name = line[4:-1]
                # 0 is background
                id_class[i + 1] = {"ycb_id": ycb_id, "name": name}
        return id_class


    def load_training_roidbs(self, names):
        """
        Args:
            names (list[str]): name of the training datasets, e.g.  ['train2014', 'valminusminival2014']

        Returns:
            roidbs (list[dict]):

        Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
        and the following keys are expected for training:

        file_name: str, full path to the image
        boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
        class: numpy array of k integers, in the range of [1, #categories], NOT [0, #categories)
        is_crowd: k booleans. Use k False if you don't know what it means.
        segmentation: k lists of numpy arrays (one for each instance).
            Each list of numpy arrays corresponds to the mask for one instance.
            Each numpy array in the list is a polygon of shape Nx2,
            because one mask can be represented by N polygons.

            If your segmentation annotations are originally masks rather than polygons,
            either convert it, or the augmentation will need to be changed or skipped accordingly.

            Include this field only if training Mask R-CNN.
        """
        return COCODetection.load_many(
            cfg.DATA.BASEDIR, names, add_gt=True, add_mask=cfg.MODE_MASK)

    def load_inference_roidbs(self, name):
        """
        Args:
            name (str): name of one inference dataset, e.g. 'minival2014'

        Returns:
            roidbs (list[dict]):

            Each dict corresponds to one image to run inference on. The
            following keys in the dict are expected:

            file_name (str): full path to the image
            image_id (str): an id for the image. The inference results will be stored with this id.
        """
        return COCODetection.load_many(cfg.DATA.BASEDIR, name, add_gt=False)

    def eval_or_save_inference_results(self, results, dataset, output=None):
        """
        Args:
            results (list[dict]): the inference results as dicts.
                Each dict corresponds to one __instance__. It contains the following keys:

                image_id (str): the id that matches `load_inference_roidbs`.
                category_id (int): the category prediction, in range [1, #category]
                bbox (list[float]): x1, y1, x2, y2
                score (float):
                segmentation: the segmentation mask in COCO's rle format.

            dataset (str): the name of the dataset to evaluate.
            output (str): the output file to optionally save the results to.

        Returns:
            dict: the evaluation results.
        """
        continuous_id_to_COCO_id = {v: k for k, v in COCODetection.COCO_id_to_category_id.items()}
        for res in results:
            # convert to COCO's incontinuous category id
            res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
            # COCO expects results in xywh format
            box = res['bbox']
            box[2] -= box[0]
            box[3] -= box[1]
            res['bbox'] = [round(float(x), 3) for x in box]

        assert output is not None, "COCO evaluation requires an output file!"
        with open(output, 'w') as f:
            json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            return COCODetection(cfg.DATA.BASEDIR, dataset).print_coco_metrics(output)
        else:
            return {}

    # code for singleton:
    _instance = None

    def file_path_from_id(self, image_id, add_str):
        """

        :param image_id: Id of the Image defined in the Dataset Files
        :param add_str: Part of the file that is added to the base path
        :return: absolute file path of the image
        """
        base_path = os.path.abspath(self.image_dir)
        full_path = os.path.join(base_path, image_id + f"-{add_str}")
        return full_path

    def load_single_roidb(self, image_id):
        image = self.load_image(image_id)
        depth = self.load_depth(image_id)
        bbox = self.load_box(image_id)
        meta = self.load_meta(image_id)
        # mask is here a boolean array of masks
        mask = self.load_mask(image_id)

    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_path = self.file_path_from_id(image_id, "color.png")
        image = cv2.imread(image_path)
        if image.ndim != 3:
            raise ImportError("Imported Image has the wrong number of dimensions.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, image_id):
        """
        Loads the depth image
        :param image_id: ID of the image to load
        :return: [H, W, 1] Image with depth values in m
        """
        depth_path = self.file_path_from_id(image_id, "depth.png")
        depth = cv2.imread(depth_path, -1)
        scaled_depth = np.expand_dims(depth / 10000., 2)
        return scaled_depth

    def load_box(self, image_id):
        """
        Loads the bounding boxes of the corresponding classes
        :param image_id:
        :return: [k, 4] array with:
            [k, 0] = x_0
            [k, 1] = y_0
            [k, 2] = x_1
            [k, 3] = y_1
        """
        meta = self.load_meta(image_id)
        category_ids = meta["cls_indexes"]
        idx = np.argsort(category_ids)
        sorted_cat_ids = category_ids[idx]
        box_path = self.file_path_from_id(image_id, "box.txt")
        box_dict = {}
        with open(box_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.split()
                cat_name = parts[0][4:-1]
                cat_id = self.category_name_to_category_id[cat_name]
                box = np.array(parts[1:], dtype=np.float32)
                box_dict[cat_id] = box
        bboxs = []
        for key in sorted(box_dict.keys()):
            bboxs.append(box_dict[key])
        return np.array(bboxs)

    def load_meta(self, image_id):
        """
        Loads the meta.mat file for each image
        :param image_id:
        :return: dict with
            center: 2D location of the projection of the 3D model origin in the image
            cls_indexes: class labels of the objects
            factor_depth: divde the depth image by this factor to get the actual depth vaule
            intrinsic_matrix: camera intrinsics
            poses: 6D poses of objects in the image
            rotation_translation_matrix: RT of the camera motion in 3D
            vertmap: coordinates in the 3D model space of each pixel in the image

        """
        path = self.file_path_from_id(image_id, "meta.mat")
        meta = scio.loadmat(path)
        return meta


    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

if __name__ == '__main__':
    cfg.DATA.BASEDIR = os.path.expanduser("~/Hitachi/YCB_Video_Dataset/")