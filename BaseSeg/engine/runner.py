
from datetime import timedelta

import torch

from BaseSeg.engine.launch import launch
from BaseSeg.engine.seg_model import SegTrainModel, SegSemiSupervisedTrainModel


class SegTrainRunner(object):
    def __init__(self, cfg):
        super(SegTrainRunner, self).__init__()
        self.cfg = cfg

        self._build_parameters()
        self._build_models()

    def _build_parameters(self):
        self.world_size = min(self.cfg.env.num_gpu, torch.cuda.device_count())
        self.dist_url = self.cfg.env.dist_url
        self.default_timeout = timedelta(minutes=30)

    def _build_models(self):
        if self.cfg.model.semi_supervised:
            self.model = SegSemiSupervisedTrainModel(self.cfg, model_name='seg_train')
        else:
            self.model = SegTrainModel(self.cfg, model_name='seg_train')

    def run(self):
        launch(self.model.run,
               self.world_size,
               is_distributed=self.cfg.env.is_distributed_train,
               dist_url=self.dist_url,
               args=(),
               timeout=self.default_timeout)























