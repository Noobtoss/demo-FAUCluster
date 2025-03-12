import argparse
from argparse import Namespace
import yaml
from ultralytics import YOLO


def train(cfg: dict):
    model = YOLO(cfg.pop("model"))
    model.train(
        **cfg
    )

def update_cfg(cfg: dict, args: Namespace) -> dict:
    for key, value in vars(args).items():
        if key != "cfg":
            if value is not None and value != "None":
                if key in cfg:
                    cfg[key] = value
                else:
                    print(f"Warning: Configuration does not have attribute '{key}'.")
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
    train(cfg)

if __name__ == '__main__':
    main()
