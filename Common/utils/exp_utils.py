

import os
import subprocess
import importlib.util


def prep_exp(dataset_path, exp_path, server_env, use_stored_settings=True, is_training=True):
    """
    I/O handling, creating of experiment folder structure.
    Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code.
    Thus, training/inference of this experiment can be started at anytime.
    Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param server_env: boolean flag. pass to configs script for cloud deployment.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing
     experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return:
    """

    if is_training:

        # the first process of an experiment creates the directories and copies the config to exp_path.
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
            os.mkdir(os.path.join(exp_path, 'plots'))
            subprocess.call('cp {} {}'.format(os.path.join(dataset_path, 'configs.py'),
                                              os.path.join(exp_path, 'configs.py')), shell=True)
            subprocess.call('cp {} {}'.format('default_configs.py',
                                              os.path.join(exp_path, 'default_configs.py')), shell=True)

        if use_stored_settings:
            subprocess.call('cp {} {}'.format('default_configs.py',
                                              os.path.join(exp_path, 'default_configs.py')), shell=True)
            cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            # only the first process copies the model selected in configs to exp_path.
            if not os.path.isfile(os.path.join(exp_path, 'model.py')):
                subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
                subprocess.call('cp {} {}'.format(os.path.join(cf.backbone_path),
                                                  os.path.join(exp_path, 'backbone.py')), shell=True)

            # copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
            tmp_model_path = os.path.join(cf.source_dir, 'models', 'tmp_model.py')
            tmp_backbone_path = os.path.join(cf.source_dir, 'models', 'tmp_backbone.py')
            subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'model.py'), tmp_model_path), shell=True)
            subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'backbone.py'), tmp_backbone_path), shell=True)
            cf.model_path = tmp_model_path
            cf.backbone_path = tmp_backbone_path

        else:
            # run training with source code info and copy snapshot of model to exp_dir for later testing
            # (overwrite scripts if exp_dir already exists.)
            cf_file = import_module('cf', os.path.join(dataset_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
            subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_path, 'backbone.py')), shell=True)
            subprocess.call('cp {} {}'.format('default_configs.py',
                                              os.path.join(exp_path, 'default_configs.py')), shell=True)
            subprocess.call('cp {} {}'.format(os.path.join(dataset_path, 'configs.py'),
                                              os.path.join(exp_path, 'configs.py')), shell=True)

    else:
        # for testing, copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
        cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
        cf = cf_file.configs(server_env)
        tmp_model_path = os.path.join(cf.source_dir, 'models', 'tmp_model.py')
        tmp_backbone_path = os.path.join(cf.source_dir, 'models', 'tmp_backbone.py')
        subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'model.py'), tmp_model_path), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_path, 'backbone.py'), tmp_backbone_path), shell=True)
        cf.model_path = tmp_model_path
        cf.backbone_path = tmp_backbone_path

    cf.exp_dir = exp_path
    cf.test_dir = os.path.join(cf.exp_dir, 'test')
    cf.plot_dir = os.path.join(cf.exp_dir, 'plots')
    cf.experiment_name = exp_path.split("/")[-1]
    cf.server_env = server_env
    cf.created_fold_id_pickle = False

    return cf


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

