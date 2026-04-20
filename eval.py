# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import multiprocessing as mp
import pprint
import yaml
import logging

from src.utils.distributed import init_distributed
from src.eval_linear_probe import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', '-c', type=str,
    help='path to config file',
    required=True)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--resume-preempt', action='store_true',
    help='resume from preempted run')
parser.add_argument(
    '--debug', type=str, default='False',
    help='run in single process to allow pdb debugging (True/False)')


def process_main(rank, config_path, world_size, devices, resume_preempt):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'config path: {config_path}')

    # -- load config params
    params = None
    with open(config_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded config...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(port=40113, rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    eval_main(args=params, resume_preempt=resume_preempt)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.debug == 'True':
        # For debugging - run single process
        process_main(0, args.config, 1, args.devices, args.resume_preempt)
    else:
      num_gpus = len(args.devices)
      mp.set_start_method('spawn')

      for rank in range(num_gpus):
          mp.Process(
              target=process_main,
              args=(rank, args.config, num_gpus, args.devices, args.resume_preempt)
          ).start()
