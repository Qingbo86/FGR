# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmagic.utils import print_colored_log

import model

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])

    cfg.visualizer.img_keys = ['pred_img']

    cfg.custom_hooks = [
        dict(type='BasicVisualizationHook', interval=1)]


    cfg.custom_test_dataloader.dataset.data_root = args.input_dir
    cfg.work_dir = args.output_dir

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()

    # print_colored_log(f'Working directory: {cfg.work_dir}')
    # print_colored_log(f'Log directory: {runner._log_dir}')

    # if args.out:

    #     class SaveMetricHook(Hook):

    #         def after_test_epoch(self, _, metrics=None):
    #             if metrics is not None:
    #                 mmengine.dump(metrics, args.out)

    #     runner.register_hook(SaveMetricHook(), 'LOWEST')

    # cfg.work_dir = osp.join(cfg.work_dir, 'val')
    # cfg.test_dataloader = cfg.val_dataloader
    # cfg.test_evaluator = cfg.val_evaluator
    # runner = Runner.from_cfg(cfg)
    # runner.test()


if __name__ == '__main__':
    main()
