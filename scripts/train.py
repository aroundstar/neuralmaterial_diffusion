from omegaconf import DictConfig
import os
import hydra

import sys
sys.path.insert(0, './')

from lib.core.trainer import Trainer
from lib.core.utils import seed_everything
from lib.main import NeuralMaterial
from lib.data import DataModule
from lib.logger import Logger
from lib.models.denoising_diffusion import Unet, GaussianDiffusion


@hydra.main(config_path="../config", config_name="default.yaml")
def train(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    seed_everything(cfg.seed)

    logger = Logger()
    model = NeuralMaterial(cfg.model, cfg.data.size)
    data = DataModule(cfg.data)

    """# diffusion model
    size = cfg.data.size
    diffusion_model = Unet(dim=64, dim_mults = (1, 2, 4, 8))
    diffusion = GaussianDiffusion(diffusion_model, image_size=size[0], timesteps=1000, loss_type='l1')"""
    
    trainer = Trainer(cfg.trainer, logger)
    trainer.fit(model, data)

if __name__ == '__main__':
    train()
