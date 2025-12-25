import hydra
from trainer.datasets.ag_news import load_ag_news
from trainer.datasets.utils import tokenize_dataset
from trainer.loop import Trainer
from trainer.models.bert import build_model
from omegaconf import DictConfig, OmegaConf
import mlflow
import random
import torch
import numpy as np
from datasets import DatasetDict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        dataset: DatasetDict = load_ag_news(cfg.dataset, subset=True)
        mlflow.log_param("train_size", len(dataset["train"]))
        assert len(dataset["train"]) != 120000, f"Dataset not loaded correctly (size {len(dataset['train'])})"

        tokenized_dataset = tokenize_dataset(cfg, dataset)
        mlflow.log_param("dataset_tokenized", True)

        model = build_model(cfg.model)
        mlflow.log_param("model_name", cfg.model.name)

        trainer = Trainer(model, tokenized_dataset, cfg)
        trainer.train()

        mlflow.pytorch.log_model(model, name="model")
