import hydra
from trainer.datasets.ag_news import load_ag_news
from trainer.models.bert import build_model
from omegaconf import DictConfig, OmegaConf
import mlflow
import random
import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)

    dataset = load_ag_news(cfg.dataset)
    mlflow.log_param("train_size", len(dataset["train"]))

    model = build_model(cfg.model)
    mlflow.log_param("model_name", cfg.model.name)

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Placeholder for training logic

        mlflow.log_metric("sanity_check", 1.0)


if __name__ == "__main__":
    main()
