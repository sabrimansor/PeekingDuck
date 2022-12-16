"""
YoloV6 Model for PeekingDuck Training

The yolov6 package is a library for implementing the YOLOv6 object detection algorithm. It provides functions for training and evaluating YOLOv6 models, as well as functions for post-processing the output of YOLOv6 models to generate bounding boxes and class predictions for objects in an image.

This package is designed to be user-friendly, with a simple API and comprehensive documentation. It is also highly customizable, allowing users to specify different parameters for the YOLOv6 algorithm and adjust the training process to their specific needs.

To use the yolov6 package, simply import it into your Python script and call the relevant functions. For example, to train a YOLOv6 model, you can use the train() function, which takes as input the training data, the model configuration, and a set of training parameters. Once the model is trained, you can use the predict() function to generate predictions on new images.

Overall, the yolov6 package provides a powerful and user-friendly tool for implementing the YOLOv6 object detection algorithm in your own projects.
"""

import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Union, Optional
from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.model.yolov6v1.yolov6_model import YoloV6Model
from peekingduck.nodes.model.yolov6v1.data.data_augment import letterbox
from peekingduck.nodes.model.yolov6v1.utils.nms import non_max_suppression
from peekingduck.nodes.model.yolov6v1.utils.events import load_yaml
from peekingduck.utils.bbox.transforms import xyxy2xyxyn

class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        conf_file = self.config["config_dir"]
        weights_file = self.config["weights_parent_dir"]
        yaml = self.config["yaml_dir"]

        self.class_names = load_yaml(yaml)['names']
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.stride = 32
        
        
        self.device = 'cpu'
        self.half = False # False

        self.model = YoloV6Model(weights_file, conf_file, device=self.device)        
        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        img_src = inputs["img"]
        img_size = inputs["img"].shape[1]

        img = self.precess_image(img_src, img_size, self.stride, self.half, self.device)        
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)[0]

        # check image and font
        assert img_src.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
        
        outputs = {"bboxes": np.zeros(shape=(1, 4)), "bbox_labels": [0], "bbox_scores": []}
        
        if len(det):
            bboxes = xyxy2xyxyn(det[:, :4], img_src.shape[0], img_src.shape[1])
            bboxes = np.array(bboxes.detach().numpy())
            labels = [ f'{self.class_names[int(cls)]} {conf:.2f}' for conf, cls in det[:, 4:6]]
            labels = np.array(labels)
            scores = np.array(det[:, 4].detach().numpy())

            outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs


    def _get_config_types(self) -> Dict[str, Any]:
        return {
            "detect": List[Union[int, str]],
            "iou_threshold": float,
            "max_output_size_per_class": int,
            "max_total_size": int,
            "model_type": str,
            "num_classes": int,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }

    @staticmethod
    def precess_image(img_src, img_size, stride, half, device):

        img = letterbox(img_src, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0

        img = img.to(device)

        if len(img.shape) == 3:
            img = img[None]

        return img