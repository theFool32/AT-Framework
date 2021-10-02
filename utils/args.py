import argparse
import time

from .configurator import Configurator
from defenses import get_defense_configuration
from datasets import get_dataset_configuration

base_config = {
    "log_step": 100,
    "batch_size": 128,
    "save_checkpoints": lambda epoch: True,
    "weight_decay": 5e-4,
    "attack_iters": 10,
    "max_epoch": 200,
    "lr_adjust": "100,150",
}


def get_configuration(config_name: str):
    dataset, lp, defense = config_name.split(":")

    dataset_config = get_dataset_configuration(dataset)[lp]
    defense_config = get_defense_configuration(defense)

    base_config.update(dataset_config)
    base_config.update(defense_config)
    base_config.update({"norm": lp})

    return base_config


def get_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--config", default="cifar10:linf:at", type=str)
    main_parser.add_argument("--model", default="PreActResNet18")
    main_parser.add_argument("--batch-size", type=int, default=128)
    main_parser.add_argument("--dataset", type=str)
    main_parser.add_argument("--data-dir", default="~/datasets/", type=str)
    main_parser.add_argument("--max-epoch", type=int, default=200)
    main_parser.add_argument("--defense", type=str)
    main_parser.add_argument("--log-step", type=int, default=100)
    main_parser.add_argument("--weight-decay", type=float, default=5e-4)
    main_parser.add_argument("--seed", default=0, type=int)
    main_parser.add_argument("--fname", type=str)
    main_parser.add_argument("--norm", type=str, choices=["linf", "l2"])
    main_parser.add_argument("--gpu", default="0", type=str)
    main_parser.add_argument("--attack", type=str, default="pgd")
    main_parser.add_argument("--project", default="AT-Framework", type=str)
    main_parser.add_argument("--tensorboard", action="store_true")
    main_parser.add_argument("--no-amp", action="store_true")

    main_parser.add_argument("--epsilon", type=int)
    main_parser.add_argument("--pgd-alpha", type=float)
    main_parser.add_argument("--attack-iters", type=int)

    main_parser.add_argument("--resume-checkpoint", default="", type=str)
    main_parser.add_argument("--lr", type=float)
    main_parser.add_argument("--lr-adjust", type=str)
    main_args, sub_args = main_parser.parse_known_args()

    config = vars(main_args)

    # Get default arguments
    config_name = main_args.config
    dataset, norm, defense = config_name.split(":")
    dataset = dataset if main_args.dataset is None else main_args.dataset
    norm = norm if main_args.norm is None else main_args.norm
    defense = defense if main_args.defense is None else main_args.defense
    config_name = f"{dataset}:{norm}:{defense}"

    config.update(get_configuration(config_name))
    config["dataset"] = dataset
    config["norm"] = norm
    config["defense"] = defense

    # Parse custom arguments
    sub_parser = main_parser.add_subparsers().add_parser("sub")
    for k, v in config.items():
        sub_parser.add_argument(f"--{k.replace('_', '-')}", type=type(v))
    sub_args, _ = sub_parser.parse_known_args(sub_args)
    config.update({k: v for k, v in vars(sub_args).items() if v is not None})

    config["config"] = config_name
    if config["fname"] is None:
        config["fname"] = config_name

    args = Configurator().update(config)
    args.config = config_name

    args.data_dir = f"{args.data_dir}/{args.dataset}"

    args.fname = args.fname + "_" + args.model
    args.checkpoints = args.fname + "_checkpoints"

    current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args.fname = args.fname + "/" + current_time
    args.checkpoints = args.checkpoints + "/" + current_time

    output_dir = "Outputs/"
    args.fname = output_dir + args.fname
    args.checkpoints = output_dir + args.checkpoints
    return args
