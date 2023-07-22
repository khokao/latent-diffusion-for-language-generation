import argparse
from pathlib import Path

from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-c', '--ckpt', type=str, default='checkpoints/pytorch_model_1.bin')
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())
    print(args)

    output_dir = Path(args['output'])
    cfg_path = output_dir / '.hydra' / 'config.yaml'
    ckpt_path = output_dir / args['ckpt']

    cfg = OmegaConf.load(cfg_path)

    module = __import__('ld4lg', fromlist=['LD4LGInterface'])
    interface_cls = getattr(module, 'LD4LGInterface')
    interface = interface_cls(cfg, output_dir, mode='test', ckpt_path=ckpt_path)

    interface.test()


if __name__ == '__main__':
    main()
