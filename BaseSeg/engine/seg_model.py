
import os
import sys
import datetime
import shutil
import numpy as np
from typing import List

from apex import amp
from apex.parallel import DistributedDataParallel

import cupy as cp
from cucim.skimage.morphology import remove_small_objects
from cucim.skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from ..models.build_model import get_network
from ..losses.build_loss import SegLoss
from ..losses.pyramid_consistency_loss import PyramidConsistencyLoss
from ..solver.optimizer import get_optimizer
from ..evaluation.build_metric import get_metric
from ..solver.lr_scheduler import get_lr_scheduler
from ..data.train_dataset import SegDataSet, DataLoaderX, SemiSegDataset
from ..data.sample_dataset import SampleDataset
from ..data.sampler import TwoStreamBatchSampler

import BaseSeg.data.test_time_aug as tta

from Common.utils.logger import get_logger
from Common.utils.profiling import get_time
from Common.utils.profiling import timer_decorate
from Common.transforms.image_resample import ScipyResample
from Common.transforms.image_transform import clip_and_normalize_mean_std
from Common.gpu_utils.cuda_wrapper import cuda_wrapper_model, cuda_wrapper_data
from Common.gpu_utils.distributed import get_rank, reduce_tensor
from Common.transforms.image_io import save_sitk_from_npy
from Common.transforms.mask_process import remove_small_cc, keep_topk_cc


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


class SegInferModel:
    def __init__(self, cfg=None, model_name='seg_infer'):
        self.cfg = cfg
        self._model_name = model_name
        self.raw_image_shape = None
        self.model = get_network(cfg)
        self.model = cuda_wrapper_model(self.model, cfg.is_cuda, cfg.is_fp16, mode='val')
        self._load_weight()
        self._set_requires_grad(False)
        if cfg.is_test_time_augment:
            # transforms = tta.Compose([tta.Flip(axis=[2, 3, 4]),
            #                           tta.Rotate90(k=[90, 180, 270], axis=(3, 4))])
            transforms = tta.Compose([tta.Rotate90(angles=[90], axis=(3, 4))])

            self.model = tta.SegmentationTTAWrapper(self.model, transforms)

    @property
    def model_name(self):
        return self._model_name

    def _load_weight(self):
        weight_path = self.cfg.weight_path
        if weight_path is not None and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)
            # self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)

        else:
            raise ValueError(f'{self._model_name} weight path is invalid!')

    def _set_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    @timer_decorate
    def run(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        data, _ = self.preprocess(data)
        data, _ = self.infer(data)
        data, _ = self.postprocess(data, spacing)

        return data

    @timer_decorate
    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        self.raw_image_shape = data.shape
        if not self.cfg.is_preprocess:
            data, _ = ScipyResample.resample_to_size(data, self.cfg.input_size, order=1)
            data = clip_and_normalize_mean_std(data, self.cfg.clip_window[0], self.cfg.clip_window[1])
        data = torch.from_numpy(data[np.newaxis, np.newaxis]).float()
        data = cuda_wrapper_data(data, self.cfg.is_cuda, self.cfg.is_fp16)

        return data

    @torch.no_grad()
    @timer_decorate
    def infer(self, data: torch.Tensor) -> np.ndarray:
        """
        current pipeline(moderate):   heatmap->sigmoid->gpu->cpu->resample->argmax
        optional pipeline1(most precision):  heatmap->sigmoid->gpu->cpu->resample->argmax->postprocess
        optional pipeline2(most quick):  heatmap->sigmoid->gpu->cpu->argmax->resample
        """
        data = self.model(data)
        data = data.sigmoid_()
        data = data.cpu().float()
        if self.cfg.ahead_resample:
            data = F.interpolate(data, size=self.raw_image_shape, mode='trilinear', align_corners=True)
        data = torch.squeeze(data, dim=0)

        image_size = list(data.size())
        if image_size[0] == 1:
            data = data.numpy().squeeze(axis=0)
            data = np.where(data>=0.5, 1, 0).astype(np.uint8)
        else:
            image_size[0] += 1
            new_data = torch.zeros(image_size)
            new_data[0, ] = 0.5
            new_data[1:, ] = data
            data = torch.max(new_data, dim=0, keepdim=True)[1].squeeze(dim=0).numpy()
            del new_data

        return data.astype(np.uint8)

    @timer_decorate
    def postprocess(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        area_least = self.cfg.area_least / spacing[0] / spacing[1] / spacing[2]

        if self.cfg.is_remove_small_objects:
            # data = remove_small_cc(data, area_least, self.cfg.keep_topk, out_mask)
            out_mask = np.zeros(data.shape, np.uint8)
            for i in range(1, self.cfg.num_class+1):
                mask = np.where(data == i, 1, 0)
                keep_topk_cc(mask, area_least, self.cfg.keep_topk, i, out_mask)
            data = out_mask.copy()

            # if self.cfg.num_class == 1:
            #     data = cp.asarray(data).astype(cp.uint8)
            #     data = remove_small_objects(data, min_size=area_least, connectivity=3, in_place=True)
            # else:
            #     mask = data.copy()
            #     end_idx = np.max(mask)
            #     mask[mask == 0] = 30
            #     start_idx = np.min(mask)
            #
            #     data = cp.asarray(data).astype(cp.uint8)
            #     for idx in range(start_idx, end_idx+1):
            #         mask = data.copy()
            #         mask = cp.where(mask == idx, cp.array(1), cp.array(0))
            #         if cp.sum(mask) > area_least:
            #             mask = remove_small_objects(mask, min_size=area_least, connectivity=3, in_place=True)
            #             data[data == idx] = 0
            #             data[mask != 0] = idx
            # data = cp.asnumpy(data)

        if not self.cfg.ahead_resample and self.cfg.delay_resample:
            data = ScipyResample.nearest_resample_mask(data, self.raw_image_shape)

        return data


class SegInferModelV1:
    def __init__(self, cfg=None, model_name='seg_infer'):
        self.cfg = cfg
        self._model_name = model_name
        self.raw_image_shape = None
        self.model = get_network(cfg)
        self.model = cuda_wrapper_model(self.model, cfg.is_cuda, cfg.is_fp16, mode='val')
        self._load_weight()
        self._set_requires_grad(False)
        if cfg.is_test_time_augment:
            transforms = tta.Compose([
                tta.Crop(crop_mode=['crop_ccc', 'crop_zzz']),
                tta.Rotate90(angles=[0, 180], axis=(3, 4)),
                # tta.Flip(axis=[3]),
            ])
            # transforms = tta.Compose([
            #     tta.Flip(axis=[3])])
            # transforms = tta.Compose([
            #     tta.Crop(crop_mode=['crop_ccc', 'crop_zzz'])])

            self.model = tta.SegmentationTTAWrapper(self.model, transforms, num_class=self.cfg.num_class, merge_mode='mean')

    @property
    def model_name(self):
        return self._model_name

    def _load_weight(self):
        weight_path = self.cfg.weight_path
        if weight_path is not None and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)

            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)

        else:
            raise ValueError(f'{self._model_name} weight path is invalid!')

    def _set_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    @timer_decorate
    def run(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        data, _ = self.preprocess(data)
        data, _ = self.infer(data)
        data, _ = self.postprocess(data, spacing)

        return data

    @timer_decorate
    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        self.raw_image_shape = data.shape
        if not self.cfg.is_preprocess:
            data, _ = ScipyResample.resample_to_size(data, self.cfg.input_size, order=1)
            data = clip_and_normalize_mean_std(data, self.cfg.clip_window[0], self.cfg.clip_window[1])
        data = torch.from_numpy(data[np.newaxis, np.newaxis]).float()
        data = cuda_wrapper_data(data, self.cfg.is_cuda, self.cfg.is_fp16)

        return data

    @torch.no_grad()
    @timer_decorate
    def infer(self, data: torch.Tensor) -> np.ndarray:
        """
        current pipeline(moderate):   heatmap->sigmoid->gpu->cpu->resample->argmax
        optional pipeline1(most precision):  heatmap->sigmoid->gpu->cpu->resample->argmax->postprocess
        optional pipeline2(most quick):  heatmap->sigmoid->gpu->cpu->argmax->resample
        """
        data = self.model(data)
        if not self.cfg.is_test_time_augment:
            data = data.sigmoid_()
            data = data.cpu().float()
        if self.cfg.ahead_resample:
            data = F.interpolate(data, size=self.raw_image_shape, mode='trilinear', align_corners=True)
        data = torch.squeeze(data, dim=0).numpy()

        return data

    @timer_decorate
    def postprocess(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        area_least = self.cfg.area_least / spacing[0] / spacing[1] / spacing[2]

        if self.cfg.is_remove_small_objects:
            # data = remove_small_cc(data, area_least, self.cfg.keep_topk, out_mask)
            out_mask = np.zeros(data.shape[1:], np.uint8)
            for i in range(1, self.cfg.num_class+1):
                mask = np.where(data[i-1] >= 0.5, 1, 0).astype(np.uint8)
                keep_topk_cc(mask, area_least, self.cfg.keep_topk, i, out_mask)
            data = out_mask.copy()

        if not self.cfg.ahead_resample and self.cfg.delay_resample:
            data = ScipyResample.nearest_resample_mask(data, self.raw_image_shape)

        return data


class SegInferModelV2:
    def __init__(self, cfg=None, model_name='seg_infer'):
        self.cfg = cfg
        self._model_name = model_name
        self.raw_image_shape = None
        self.model = get_network(cfg)
        self.model = cuda_wrapper_model(self.model, cfg.is_cuda, cfg.is_fp16, mode='val')
        self._load_weight()
        self._set_requires_grad(False)
        if cfg.is_test_time_augment:
            transforms = tta.Compose([
                tta.Crop(crop_mode=['crop_ccc', 'crop_zzz']),
                tta.Rotate90(angles=[0, 180], axis=(3, 4)),
                # tta.Flip(axis=[3]),
            ])

            self.model = tta.SegmentationTTAWrapper(self.model, transforms, num_class=self.cfg.num_class, merge_mode='mean')

    @property
    def model_name(self):
        return self._model_name

    def _load_weight(self):
        weight_path = self.cfg.weight_path
        if weight_path is not None and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)

            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)

        else:
            raise ValueError(f'{self._model_name} weight path is invalid!')

    def _set_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    @timer_decorate
    def run(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        data, _ = self.preprocess(data)
        data, _ = self.infer(data)
        data, _ = self.postprocess(data, spacing)

        return data

    @timer_decorate
    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        self.raw_image_shape = data.shape
        if not self.cfg.is_preprocess:
            data, _ = ScipyResample.resample_to_size(data, self.cfg.input_size, order=1)
            data = clip_and_normalize_mean_std(data, self.cfg.clip_window[0], self.cfg.clip_window[1])
        data = torch.from_numpy(data[np.newaxis, np.newaxis]).float()
        data = cuda_wrapper_data(data, self.cfg.is_cuda, self.cfg.is_fp16)

        return data

    @torch.no_grad()
    @timer_decorate
    def infer(self, data: torch.Tensor) -> np.ndarray:
        """
        current pipeline(moderate):   heatmap->sigmoid->gpu->cpu->resample->argmax
        optional pipeline1(most precision):  heatmap->sigmoid->gpu->cpu->resample->argmax->postprocess
        optional pipeline2(most quick):  heatmap->sigmoid->gpu->cpu->argmax->resample
        """
        data = self.model(data)
        if not self.cfg.is_test_time_augment:
            data = data.sigmoid_()
            data = data.cpu().float()
        if self.cfg.ahead_resample:
            data = F.interpolate(data, size=self.raw_image_shape, mode='trilinear', align_corners=True)

        if self.cfg.keep_topk == 1:
            data = torch.squeeze(data, dim=0).numpy()

            return data
        else:
            data = torch.squeeze(data, dim=0)

            image_size = list(data.size())
            if image_size[0] == 1:
                data = data.numpy().squeeze(axis=0)
                data = np.where(data >= 0.5, 1, 0).astype(np.uint8)
            else:
                image_size[0] += 1
                new_data = torch.zeros(image_size)
                new_data[0,] = 0.5
                new_data[1:, ] = data
                data = torch.max(new_data, dim=0, keepdim=True)[1].squeeze(dim=0).numpy()
                del new_data

            return data.astype(np.uint8)

    @timer_decorate
    def postprocess(self, data: np.ndarray, spacing: List[float]) -> np.ndarray:
        area_least = self.cfg.area_least / spacing[0] / spacing[1] / spacing[2]

        if self.cfg.is_remove_small_objects:
            if self.cfg.keep_topk == 1:
                out_mask = np.zeros(data.shape[1:], np.uint8)
                for i in range(1, self.cfg.num_class + 1):
                    mask = np.where(data[i - 1] >= 0.5, 1, 0).astype(np.uint8)
                    keep_topk_cc(mask, area_least, self.cfg.keep_topk, i, out_mask)
                data = out_mask.copy()
            else:
                out_mask = np.zeros(data.shape, np.uint8)
                data = remove_small_cc(data, area_least, self.cfg.keep_topk, out_mask)

        if not self.cfg.ahead_resample and self.cfg.delay_resample:
            data = ScipyResample.nearest_resample_mask(data, self.raw_image_shape)

        return data


class SegTrainModel:
    def __init__(self, cfg, model_name='seg_train'):
        self.cfg = cfg
        self._model_name = model_name
        self._is_build = False

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_main_process(self):
        return self.local_rank == 0

    def _build(self):
        self.local_rank = get_rank()
        self.world_size = torch.cuda.device_count()

        self._build_parameters()
        self._build_writes()
        self._build_dataset()

        self._build_network()
        self._build_solver()
        self._load_checkpoint()
        self._build_ddp_apex_training()
        self._build_loss()
        self._build_metric()

        self._is_build = True

    def _build_parameters(self):
        self.start_epoch = self.cfg.slover.start_epoch
        self.total_epoch = 10 if self.cfg.env.smoke_test else self.cfg.slover.total_epoch
        self.lr = self.cfg.slover.lr
        self.best_dice = 0

        if torch.cuda.is_available():
            self.is_cuda = True
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.is_cuda = False
            self.device = torch.device('cpu', self.local_rank)

        self.is_apex_train = self.cfg.env.is_apex_train if self.is_cuda else False
        self.is_distributed_train = self.cfg.env.is_distributed_train if self.is_cuda and \
                                    self.world_size > 1 else False

    def _build_writes(self):
        self.base_save_dir = os.path.join(self.cfg.env.save_dir,
                                          self.cfg.env.exp_name + '_result'
                                          '_time-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if self.is_main_process and not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

        # copy codes
        self.code_save_dir = os.path.join(self.base_save_dir, 'codes')
        if self.is_main_process:
            if not os.path.exists(self.code_save_dir):
                os.makedirs(self.code_save_dir)
            shutil.copytree(BASE_DIR+'/BaseSeg', self.code_save_dir+'/BaseSeg')
            shutil.copytree(BASE_DIR+'/Common', self.code_save_dir+'/Common')
            if self.cfg.data_loader.semi_dataset:
                shutil.copy('./semi_config.yaml', self.code_save_dir + '/semi_config.yaml')
            else:
                shutil.copy('./full_config.yaml', self.code_save_dir+'/full_config.yaml')
            shutil.copy('./run.py', self.code_save_dir+'/run.py')

        # log dir
        self.log_dir = os.path.join(self.base_save_dir, 'logs')
        if self.is_main_process and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.is_main_process:
            self.logger = get_logger(self.log_dir)
            self.logger.info('\n------------ train options -------------')
            self.logger.info(str(self.cfg.pretty_text))
            self.logger.info('-------------- End ----------------\n')

        # model dir
        self.model_dir = os.path.join(self.base_save_dir, 'models')
        if self.is_main_process and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.save_weight_path = os.path.join(self.model_dir, 'best_model.pt')
        self.pretrain_model_path = self.cfg.model.weight_path

        # tensorboard writer
        if self.is_main_process:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            self.val_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))

    def _build_dataset(self):
        if self.cfg.data_loader.semi_dataset:
            train_dataset = SemiSegDataset(self.cfg, 'train')
        else:
            train_dataset = SegDataSet(self.cfg, 'train')
        val_dataset = SegDataSet(self.cfg, 'val')

        if self.is_distributed_train:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            self.train_sampler = None
            self.val_sampler = None
        self.num_worker = self.cfg.data_loader.batch_size + 2 if self.cfg.data_loader.num_worker < \
                              self.cfg.data_loader.batch_size + 2 else self.cfg.data_loader.num_worker

        self.train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=True if self.train_sampler is None and not self.cfg.data_loader.semi_dataset else False,
            drop_last=False,
            pin_memory=True,
            sampler=self.train_sampler)
        self.val_loader = DataLoaderX(
            dataset=val_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            sampler=self.val_sampler)
        if self.is_main_process:
            self.logger.info('build data loader success!')

    def _build_network(self):
        self.model = get_network(self.cfg.model)
        self.model = self.model.to(self.device)

    def _build_solver(self):
        self.optimizer = get_optimizer(self.cfg.slover, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(self.cfg.slover, self.optimizer)

    def _build_ddp_apex_training(self):
        # set apex training
        if self.is_apex_train:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        # set distribute training
        if not self.is_distributed_train and self.world_size > 1:
            self.model = torch.nn.DataParallel(self.model)
        elif self.is_apex_train and self.is_distributed_train:
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)
        elif self.is_distributed_train:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)

    def _build_loss(self):
        self.train_loss_func = SegLoss(loss_func=self.cfg.slover.loss,
                                       loss_weight=self.cfg.slover.loss_weight,
                                       activation=self.cfg.model.activation,
                                       reduction='mean',
                                       num_label=self.cfg.model.num_class)
        self.val_loss_func = SegLoss(loss_func=self.cfg.slover.loss,
                                     loss_weight=self.cfg.slover.loss_weight,
                                     activation=self.cfg.model.activation,
                                     reduction='sum',
                                     num_label=self.cfg.model.num_class)

    def _build_metric(self):
        self.train_metric_func = get_metric(metric=self.cfg.slover.metric,
                                            activation=self.cfg.model.activation,
                                            reduction='sum')
        self.val_metric_func = get_metric(metric=self.cfg.slover.metric,
                                          activation=self.cfg.model.activation,
                                          reduction='sum')

    def run(self):
        if not self._is_build:
            self._build()
        if self.is_main_process:
            run_start_time = get_time(self.is_cuda)
            self.logger.info(f'Preprocess parallels: {self.num_worker}')
            self.logger.info(f'train samples per epoch: {len(self.train_loader)}')
            self.logger.info(f'val samples per epoch: {len(self.val_loader)}')

        for epoch in range(self.start_epoch, self.total_epoch + 1):
            if self.is_main_process:
                self.logger.info(f'\nStarting training epoch {epoch}')

            epoch_start_time = get_time(self.is_cuda)
            self.run_train_process(epoch)
            val_dice = self.run_val_process(epoch)

            self._save_checkpoint(epoch, val_dice)
            if self.is_main_process:
                self.logger.info(f'End of epoch {epoch}, time: {get_time(self.is_cuda)-epoch_start_time}')

        if self.cfg.model.is_refine:
            self.run_refine()

        if self.is_main_process:
            self.train_writer.close()
            self.val_writer.close()
            self.logger.info(f'\nEnd of training, best dice: {self.best_dice}')
            run_end_time = get_time(self.is_cuda)
            self.logger.info(f'Training time: {(run_end_time-run_start_time) / 60 / 60} hours')

    def run_refine(self):
        train_dataset = SemiSegDataset(self.cfg, 'refine')
        self.train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=True if self.train_sampler is None and not self.cfg.data_loader.semi_dataset else False,
            drop_last=False,
            pin_memory=True,
            sampler=self.train_sampler)
        for epoch in range(self.total_epoch, self.total_epoch + self.cfg.model.refine_epoch):
            if self.is_main_process:
                self.logger.info(f'\nStarting refine epoch {epoch}')

            epoch_start_time = get_time(self.is_cuda)
            self.run_train_process(epoch)
            val_dice = self.run_val_process(epoch)

            self._save_checkpoint(epoch, val_dice)
            if self.is_main_process:
                self.logger.info(f'End of epoch {epoch}, time: {get_time(self.is_cuda) - epoch_start_time}')

    @staticmethod
    def _get_lr(epoch, num_epochs, init_lr):
        if epoch <= num_epochs * 0.66:
            lr = init_lr
        elif epoch <= num_epochs * 0.86:
            lr = init_lr * 0.1
        else:
            lr = init_lr * 0.05

        return lr

    def run_train_process(self, epoch):
        self.model.train()

        train_dice = [0.] * len(self.cfg.data_loader.label_index)
        train_total = 0

        current_lr = None
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr(epoch, self.total_epoch, self.lr)
            if current_lr is None:
                current_lr = param_group['lr']

        for index, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            output_seg = self.model(images)
            seg_loss = self.train_loss_func(output_seg, masks)

            if self.is_apex_train:
                with amp.scale_loss(seg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                seg_loss.backward()

            # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)

            self.optimizer.step()
            # self.lr_scheduler.step()
            # current_lr = self.optimizer.param_groups[0]["lr"]
            dice_output = self.train_metric_func(output_seg, masks)

            if self.is_distributed_train:
                dice_output = reduce_tensor(dice_output.data)
                seg_loss = reduce_tensor(seg_loss.data)
            for i, dice_tmp in enumerate(dice_output):
                train_dice[i] += float(dice_tmp.item())
            train_total += len(images)

            if self.is_main_process:
                if index > 0 and index % self.cfg.slover.save_frequency == 0:
                    self.logger.info('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.total_epoch,
                        index * len(images), len(self.train_loader),
                        100. * index / len(self.train_loader)))
                    self.logger.info('SegLoss:{:.6f}, LearnRate:{:.6f}'.format(seg_loss.item(), current_lr))

                    for i, dice_label in enumerate(train_dice):
                        dice_ind = dice_label / train_total
                        self.logger.info('{} Dice:{:.6f}'.format(self.cfg.data_loader.label_name[i], dice_ind))

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_seg
                torch.cuda.empty_cache()

        if self.is_main_process:
            self.train_writer.add_scalar('Train/SegLoss', seg_loss.item(), epoch)
            self.train_writer.add_scalar('Train/LearnRate', current_lr, epoch)

            for i, dice_label in enumerate(train_dice):
                dice_ind = dice_label / train_total
                self.train_writer.add_scalars('Train/Dice',
                                              {self.cfg.data_loader.label_name[i]: dice_ind}, epoch)

    def run_val_process(self, epoch):
        self.model.eval()

        val_dice = [0.] * len(self.cfg.data_loader.label_index)
        val_total = 0
        val_loss = 0

        for index, (images, masks) in enumerate(self.val_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            with torch.no_grad():
                output_seg = self.model(images)

            seg_loss = self.val_loss_func(output_seg, masks)
            dice_output = self.val_metric_func(output_seg, masks)

            if self.is_distributed_train:
                seg_loss = reduce_tensor(seg_loss.data)
                dice_output = reduce_tensor(dice_output.data)
            val_loss += float(seg_loss.item())
            for i, dice_tmp in enumerate(dice_output):
                val_dice[i] += float(dice_tmp.item())
            val_total += len(images)

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_seg
                torch.cuda.empty_cache()

        val_loss /= val_total
        total_dice = 0
        if self.is_main_process:
            self.logger.info('Loss of validation is {}'.format(val_loss))
            self.val_writer.add_scalar('Val/Loss', val_loss, epoch)

            for idx, _ in enumerate(val_dice):
                val_dice[idx] /= val_total
                self.logger.info('{} Dice:{:.6f}'.format(self.cfg.data_loader.label_name[idx], val_dice[idx]))
                self.val_writer.add_scalars('Val/Dice',
                                            {self.cfg.data_loader.label_name[idx]: val_dice[idx]}, epoch)
                total_dice += val_dice[idx]
            total_dice /= len(val_dice)
            self.logger.info(f'Average dice: {total_dice}')

        return total_dice

    def _load_checkpoint(self):
        if self.is_main_process and self.pretrain_model_path is not None \
                and os.path.exists(self.pretrain_model_path):
            checkpoint = torch.load(self.pretrain_model_path)
            # self.fine_model.load_state_dict(
            #     {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)
            self.logger.info(f'load model weight success!')

    def _save_checkpoint(self, epoch, dice):
        if self.is_main_process and dice > self.best_dice:
            self.best_dice = dice
            # "lr_scheduler_dict": self.lr_scheduler.state_dict()},
            torch.save({
                'lr': self.lr,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()},
                self.save_weight_path)
        if self.is_main_process and self.cfg.slover.save_checkpoints and epoch in \
                [int(self.total_epoch*0.36), int(self.total_epoch*0.66), int(self.total_epoch*0.86), self.total_epoch]:
            save_weight_path = os.path.join(self.model_dir, f'epoch_{str(epoch)}_model.pt')
            torch.save({
                'lr': self.lr,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()},
                save_weight_path)


class SegSemiSupervisedTrainModel:
    def __init__(self, cfg, model_name='seg_train'):
        self.cfg = cfg
        self._model_name = model_name
        self._is_build = False

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_main_process(self):
        return self.local_rank == 0

    def _build(self):
        self.local_rank = get_rank()
        self.world_size = torch.cuda.device_count()

        self._build_parameters()
        self._build_writes()
        self._build_dataset()

        self._build_network()
        self._build_solver()
        self._load_checkpoint()
        self._build_ddp_apex_training()
        self._build_loss()
        self._build_metric()

        self._is_build = True

    def _build_parameters(self):
        self.start_epoch = self.cfg.slover.start_epoch
        self.total_epoch = 10 if self.cfg.env.smoke_test else self.cfg.slover.total_epoch
        self.lr = self.cfg.slover.lr
        self.best_dice = 0

        if torch.cuda.is_available():
            self.is_cuda = True
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.is_cuda = False
            self.device = torch.device('cpu', self.local_rank)

        self.is_apex_train = self.cfg.env.is_apex_train if self.is_cuda else False
        self.is_distributed_train = self.cfg.env.is_distributed_train if self.is_cuda and \
                                    self.world_size > 1 else False

    def _build_writes(self):
        self.base_save_dir = os.path.join(self.cfg.env.save_dir,
                                          self.cfg.env.exp_name + '_result'
                                          '_time-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if self.is_main_process and not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

        # copy codes
        self.code_save_dir = os.path.join(self.base_save_dir, 'codes')
        if self.is_main_process:
            if not os.path.exists(self.code_save_dir):
                os.makedirs(self.code_save_dir)
            shutil.copytree(BASE_DIR+'/BaseSeg', self.code_save_dir+'/BaseSeg')
            shutil.copytree(BASE_DIR+'/Common', self.code_save_dir+'/Common')
            shutil.copy('./config.yaml', self.code_save_dir+'/config.yaml')
            shutil.copy('./run.py', self.code_save_dir+'/run.py')

        # log dir
        self.log_dir = os.path.join(self.base_save_dir, 'logs')
        if self.is_main_process and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.is_main_process:
            self.logger = get_logger(self.log_dir)
            self.logger.info('\n------------ train options -------------')
            self.logger.info(str(self.cfg.pretty_text))
            self.logger.info('-------------- End ----------------\n')

        # model dir
        model_dir = os.path.join(self.base_save_dir, 'models')
        if self.is_main_process and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.save_weight_path = os.path.join(model_dir, 'best_model.pt')
        self.pretrain_model_path = self.cfg.model.weight_path

        # tensorboard writer
        if self.is_main_process:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            self.val_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))

    def _build_dataset(self):
        train_dataset = SampleDataset(self.cfg, 'train')
        val_dataset = SampleDataset(self.cfg, 'val')

        self.num_worker = self.cfg.data_loader.batch_size + 2 if self.cfg.data_loader.num_worker < \
                          self.cfg.data_loader.batch_size + 2 else self.cfg.data_loader.num_worker

        if self.is_distributed_train:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.train_loader = DataLoaderX(
                dataset=train_dataset,
                batch_size=self.cfg.data_loader.batch_size,
                num_workers=self.num_worker,
                shuffle=True if train_sampler is None else False,
                drop_last=False,
                pin_memory=True,
                sampler=train_sampler)
        else:
            val_sampler = None
            labeled_idxs = list(range(0, train_dataset.num_labeled))
            unlabeled_idxs = list(range(train_dataset.num_labeled, train_dataset.num_total))
            train_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                                  self.cfg.data_loader.batch_size,
                                                  self.cfg.data_loader.batch_size - self.cfg.data_loader.labeled_bs)
            self.train_loader = DataLoaderX(
                dataset=train_dataset,
                num_workers=self.num_worker,
                pin_memory=True,
                batch_sampler=train_sampler)

            # weights = np.array(weights)
            # sampler_weights = torch.from_numpy(weights).double()
            # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights,
            #                                                                int(weights.shape[0] * 1))

        self.val_loader = DataLoaderX(
            dataset=val_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            sampler=val_sampler)
        if self.is_main_process:
            self.logger.info('build data loader success!')

    def _build_network(self):
        self.model = get_network(self.cfg.model)
        self.model = self.model.to(self.device)

    def _build_solver(self):
        self.optimizer = get_optimizer(self.cfg.slover, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(self.cfg.slover, self.optimizer)

    def _build_ddp_apex_training(self):
        # set apex training
        if self.is_apex_train:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        # set distribute training
        if not self.is_distributed_train and self.world_size > 1:
            self.model = torch.nn.DataParallel(self.model)
        elif self.is_apex_train and self.is_distributed_train:
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)
        elif self.is_distributed_train:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)

    def _build_loss(self):
        self.train_loss_func = SegLoss(loss_func=self.cfg.slover.loss,
                                       loss_weight=self.cfg.slover.loss_weight,
                                       activation=self.cfg.model.activation,
                                       reduction='mean',
                                       num_label=self.cfg.model.num_class)
        self.val_loss_func = SegLoss(loss_func=self.cfg.slover.loss,
                                     loss_weight=self.cfg.slover.loss_weight,
                                     activation=self.cfg.model.activation,
                                     reduction='sum',
                                     num_label=self.cfg.model.num_class)
        self.semi_supervised_loss = PyramidConsistencyLoss(self.cfg.data_loader.labeled_bs)

    def _build_metric(self):
        self.train_metric_func = get_metric(metric=self.cfg.slover.metric,
                                            activation=self.cfg.model.activation,
                                            reduction='sum')
        self.val_metric_func = get_metric(metric=self.cfg.slover.metric,
                                          activation=self.cfg.model.activation,
                                          reduction='sum')

    def run(self):
        if not self._is_build:
            self._build()
        if self.is_main_process:
            run_start_time = get_time(self.is_cuda)
            self.logger.info(f'Preprocess parallels: {self.num_worker}')
            self.logger.info(f'train samples per epoch: {len(self.train_loader)}')
            self.logger.info(f'val samples per epoch: {len(self.val_loader)}')

        for epoch in range(self.start_epoch, self.total_epoch + 1):
            if self.is_main_process:
                self.logger.info(f'\nStarting training epoch {epoch}')

            epoch_start_time = get_time(self.is_cuda)
            self.run_train_process(epoch)
            val_dice = self.run_val_process(epoch)

            self._save_checkpoint(epoch, val_dice)
            if self.is_main_process:
                self.logger.info(f'End of epoch {epoch}, time: {get_time(self.is_cuda)-epoch_start_time}')

        if self.is_main_process:
            self.train_writer.close()
            self.val_writer.close()
            self.logger.info(f'\nEnd of training, best dice: {self.best_dice}')
            run_end_time = get_time(self.is_cuda)
            self.logger.info(f'Training time: {(run_end_time-run_start_time) / 60 / 60} hours')

    @staticmethod
    def _get_lr(epoch, num_epochs, init_lr):
        if epoch <= num_epochs * 0.66:
            lr = init_lr
        elif epoch <= num_epochs * 0.86:
            lr = init_lr * 0.1
        else:
            lr = init_lr * 0.05

        return lr

    def run_train_process(self, epoch):
        self.model.train()

        train_dice = [0.] * len(self.cfg.data_loader.label_index)
        train_total = 0

        current_lr = None
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr(epoch, self.total_epoch, self.lr)
            if current_lr is None:
                current_lr = param_group['lr']

        for index, sampled_batch in enumerate(self.train_loader):
            images, masks = sampled_batch['image'], sampled_batch['onehot_label']
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            output_list = self.model(images)
            num_output = len(output_list)
            seg_loss = 0
            for i in range(num_output):
                loss = self.train_loss_func(output_list[i][:self.cfg.data_loader.labeled_bs],
                                            masks[:self.cfg.data_loader.labeled_bs])
                seg_loss += loss
                del loss
                torch.cuda.empty_cache()
            seg_loss /= num_output

            if self.cfg.model.semi_supervised and num_output > 1:
                consistency_loss = self.semi_supervised_loss(output_list)
                seg_loss += 0.3 * consistency_loss

            if self.is_apex_train:
                with amp.scale_loss(seg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                seg_loss.backward()

            # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            dice_output = self.train_metric_func(output_list[-1][:self.cfg.data_loader.labeled_bs],
                                                 masks[:self.cfg.data_loader.labeled_bs])

            if self.is_distributed_train:
                dice_output = reduce_tensor(dice_output.data)
                seg_loss = reduce_tensor(seg_loss.data)
            for i, dice_tmp in enumerate(dice_output):
                train_dice[i] += float(dice_tmp.item())
            train_total += len(images)

            if self.is_main_process:
                if index > 0 and index % self.cfg.slover.save_frequency == 0:
                    self.logger.info('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.total_epoch,
                        index * len(images), len(self.train_loader),
                        100. * index / len(self.train_loader)))
                    self.logger.info('SegLoss:{:.6f}, LearnRate:{:.6f}'.format(seg_loss.item(), current_lr))

                    for i, dice_label in enumerate(train_dice):
                        dice_ind = dice_label / train_total
                        self.logger.info('{} Dice:{:.6f}'.format(self.cfg.data_loader.label_name[i], dice_ind))

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_list
                torch.cuda.empty_cache()

        if self.is_main_process:
            self.train_writer.add_scalar('Train/SegLoss', seg_loss.item(), epoch)
            self.train_writer.add_scalar('Train/LearnRate', current_lr, epoch)

            for i, dice_label in enumerate(train_dice):
                dice_ind = dice_label / train_total
                self.train_writer.add_scalars('Train/Dice',
                                              {self.cfg.data_loader.label_name[i]: dice_ind}, epoch)

    def run_val_process(self, epoch):
        self.model.eval()

        val_dice = [0.] * len(self.cfg.data_loader.label_index)
        val_total = 0
        val_loss = 0

        for images, masks in self.val_loader:
            images, masks = images.to(self.device), masks.to(self.device)

            with torch.no_grad():
                output_list = self.model(images)

            seg_loss = self.val_loss_func(output_list[-1], masks)
            dice_output = self.val_metric_func(output_list[-1], masks)

            if self.is_distributed_train:
                seg_loss = reduce_tensor(seg_loss.data)
                dice_output = reduce_tensor(dice_output.data)
            val_loss += float(seg_loss.item())
            for i, dice_tmp in enumerate(dice_output):
                val_dice[i] += float(dice_tmp.item())
            val_total += len(images)

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_list
                torch.cuda.empty_cache()

        val_loss /= val_total
        total_dice = 0
        if self.is_main_process:
            self.logger.info('Loss of validation is {}'.format(val_loss))
            self.val_writer.add_scalar('Val/Loss', val_loss, epoch)

            for idx, _ in enumerate(val_dice):
                val_dice[idx] /= val_total
                self.logger.info('{} Dice:{:.6f}'.format(self.cfg.data_loader.label_name[idx], val_dice[idx]))
                self.val_writer.add_scalars('Val/Dice',
                                            {self.cfg.data_loader.label_name[idx]: val_dice[idx]}, epoch)
                total_dice += val_dice[idx]
            total_dice /= len(val_dice)
            self.logger.info(f'Average dice: {total_dice}')

        return total_dice

    def _load_checkpoint(self):
        if self.is_main_process and self.pretrain_model_path is not None \
                and os.path.exists(self.pretrain_model_path):
            checkpoint = torch.load(self.pretrain_model_path)
            # self.fine_model.load_state_dict(
            #     {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)
            self.logger.info(f'load model weight success!')

    def _save_checkpoint(self, epoch, dice):
        if self.is_main_process and dice > self.best_dice:
            self.best_dice = dice
            # "lr_scheduler_dict": self.lr_scheduler.state_dict()},
            torch.save({
                'lr': self.lr,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()},
                self.save_weight_path)

