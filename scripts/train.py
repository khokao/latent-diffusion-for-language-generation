from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ld4lg.cfg import MyConfig

OmegaConf.register_new_resolver('eval', eval)


@hydra.main(version_base=None, config_path='../ld4lg/cfg', config_name='config')
def main(cfg: MyConfig):
    output_dir = Path(HydraConfig.get().run.dir)
    assert output_dir.exists()

    module = __import__('ld4lg', fromlist=['LD4LGInterface'])
    interface_cls = getattr(module, 'LD4LGInterface')
    interface = interface_cls(cfg, output_dir, mode='train')

    interface.train()


if __name__ == '__main__':
    main()
