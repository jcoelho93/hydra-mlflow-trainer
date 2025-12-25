from datasets import load_dataset


def load_ag_news(cfg):
    dataset = load_dataset(cfg.name)
    return dataset
