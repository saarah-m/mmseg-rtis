# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    parser.add_argument(
        '--show-time-per-frame', action='store_true',
        help='show time per frame in addition to FPS')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmseg'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []
    overall_time_per_frame_list = []
    cfg.test_dataloader.batch_size = 1
    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        try:
            # build the dataloader
            data_loader = Runner.build_dataloader(cfg.test_dataloader)

            # build the model and load checkpoint
            cfg.model.train_cfg = None
            model = MODELS.build(cfg.model)

            if 'checkpoint' in args and osp.exists(args.checkpoint):
                load_checkpoint(model, args.checkpoint, map_location='cpu')

            if torch.cuda.is_available():
                model = model.cuda()

            model = revert_sync_batchnorm(model)
            model.eval()

            # the first several iterations may be very slow so skip them
            num_warmup = 5
            pure_inf_time = 0
            total_iters = 200

            # benchmark with 200 batches and take the average
            for i, data in enumerate(data_loader):
                data = model.data_preprocessor(data, True)
                inputs = data['inputs']
                data_samples = data['data_samples']

                # Skip if input has zero dimension
                if inputs.shape[-2] == 0 or inputs.shape[-1] == 0:
                    raise ValueError(f"Invalid input dimensions: {inputs.shape}")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    model(inputs, data_samples, mode='predict')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                if i >= num_warmup:
                    pure_inf_time += elapsed
                    if (i + 1) % args.log_interval == 0:
                        fps = (i + 1 - num_warmup) / pure_inf_time
                        time_per_frame = pure_inf_time / (i + 1 - num_warmup)
                        if args.show_time_per_frame:
                            print(f'Done image [{i + 1:<3}/ {total_iters}], '
                                  f'fps: {fps:.2f} img / s, '
                                  f'time per frame: {time_per_frame:.4f}s')
                        else:
                            print(f'Done image [{i + 1:<3}/ {total_iters}], '
                                  f'fps: {fps:.2f} img / s')

                if (i + 1) == total_iters:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    time_per_frame = pure_inf_time / (i + 1 - num_warmup)
                    if args.show_time_per_frame:
                        print(f'Overall fps: {fps:.2f} img / s, '
                              f'time per frame: {time_per_frame:.4f}s\n')
                    else:
                        print(f'Overall fps: {fps:.2f} img / s\n')
                    benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                    benchmark_dict[f'time_per_frame_{time_index + 1}'] = round(time_per_frame, 4)
                    overall_fps_list.append(fps)
                    overall_time_per_frame_list.append(time_per_frame)
                    break

        except Exception as e:
            print(f"Error during benchmark run {time_index + 1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            return

    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    if args.show_time_per_frame:
        benchmark_dict['average_time_per_frame'] = round(np.mean(overall_time_per_frame_list), 4)
        benchmark_dict['time_per_frame_variance'] = round(np.var(overall_time_per_frame_list), 6)
        print(f'Average fps of {repeat_times} evaluations: '
              f'{benchmark_dict["average_fps"]}')
        print(f'Average time per frame of {repeat_times} evaluations: '
              f'{benchmark_dict["average_time_per_frame"]}s')
        print(f'The variance of {repeat_times} evaluations: '
              f'fps: {benchmark_dict["fps_variance"]}, '
              f'time per frame: {benchmark_dict["time_per_frame_variance"]}')
    else:
        print(f'Average fps of {repeat_times} evaluations: '
              f'{benchmark_dict["average_fps"]}')
        print(f'The variance of {repeat_times} evaluations: '
              f'{benchmark_dict["fps_variance"]}')
    dump(benchmark_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
