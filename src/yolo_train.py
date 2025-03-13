import os
import argparse
from argparse import Namespace
from datetime import datetime
import yaml
from ultralytics import YOLO


def train(cfg: dict):
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)

def update_cfg(cfg: dict, args: Namespace) -> dict:
    for key, value in vars(args).items():
        if key != "cfg":
            if value is not None:
                if key in cfg:
                    cfg[key] = value
                else:
                    print(f"Warning: Configuration does not have attribute '{key}'.")
    return cfg

def update_cfg_hardcoded(cfg: dict) -> dict:
    cfg["project"] = "runs"
    model_name = os.path.splitext(os.path.basename(cfg["model"]))[0]
    data_name = os.path.splitext(os.path.basename(cfg["data"]))[0].lower()
    seed = cfg["seed"]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    cfg["name"] = f"{model_name}-{data_name}-{seed}-{timestamp}"
    return cfg

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg, "r"))
    cfg = update_cfg(cfg, args)
    cfg = update_cfg_hardcoded(cfg)
    train(cfg)

if __name__ == '__main__':
    main()
