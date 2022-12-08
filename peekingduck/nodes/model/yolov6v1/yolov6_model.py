
import torch, torchvision
import numpy as np
from pathlib import Path

from peekingduck.nodes.model.yolov6v1.utils.events import LOGGER
from peekingduck.nodes.model.yolov6v1.models import yolo
from peekingduck.nodes.model.yolov6v1.utils.config import Config

class YoloV6Model(torch.nn.Module):
    def __init__(self, weights='weights/yolov6n.pt', conf_file='configs/yolov6n.py',  device=None, dnn=True):

        super().__init__()
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} format is not supported.'
        model = self.load_checkpoint(weights, conf_file, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


    def load_checkpoint(self, weights, conf_file, map_location=None, inplace=True, fuse=True):
        """Load model from checkpoint file."""
        LOGGER.info("Loading checkpoint from {}".format(weights))
        
        cfg = Config.fromfile(conf_file)
        setattr(cfg, 'training_mode', 'repvgg')
        model = yolo.build_model(cfg, 80, map_location)

        ckpt = torch.load(str(weights), map_location=map_location)
        # model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        model.load_state_dict(ckpt, strict=False)

        if fuse:
            from peekingduck.nodes.model.yolov6v1.utils.torch_utils import fuse_model
            LOGGER.info("\nFusing model...")
            model = fuse_model(model).eval()
        else:
            model = model.eval()
        return model

