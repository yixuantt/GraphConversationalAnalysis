import yaml
import argparse

import os
import logging
import torch
from ast import literal_eval

from trainer import *
from utils.utils import prepare_device
import numpy as np

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seed', type=int, default=7890, help='Random seed')
    parser.add_argument('--trainer', type=str,
                        default='GCATrainer', help='The trainer to execute')
    parser.add_argument(
        '--config', type=str, default='topic_profet_gnn.yml', help='Path to the config file')
    parser.add_argument('--verbose', type=str, default='info',
                        help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--doc', type=str, default='GCATrainer',
                        help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='',
                        help='A string for experiment comment')
    parser.add_argument('--result', type=str, default='result',
                        help='Path for saving running related data.')
    parser.add_argument('--test', type=str, default=False,
                        help='Whether load model for test')
    parser.add_argument('--test_svr', type=str, default=False,
                        help='Whether load model for test')
    parser.add_argument('--test_interpretability', type=str, default=False,
                        help='Whether load model for test')

    args = parser.parse_args()
    args.doc = args.trainer
    args.log = os.path.join(args.result, 'log', args.doc)
    
    args.checkpoint = os.path.join(args.result, 'checkpoint', args.doc)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    with open(os.path.join('conf', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # Setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler2)
    logger.setLevel(level)

    # Add device
    device = prepare_device()
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Args = {}".format(args))
    logging.info("Config = {}".format(config))

    # Load trainer
    if args.test_interpretability:
        trainer = eval(args.trainer)(args, config)
        trainer.test_one_firm_with_interpretability()
        exit()
    if args.test_svr:
        trainer = eval(args.trainer)(args, config)
        trainer.test_svr_only()
    elif args.test:
        trainer = eval(args.trainer)(args, config)
        trainer.test(load_pre_train=True)
    else:
        trainer = eval(args.trainer)(args, config)
        trainer.train()
        trainer.test(load_pre_train=False)


if __name__ == '__main__':
    main()
