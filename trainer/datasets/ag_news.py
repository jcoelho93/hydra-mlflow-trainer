from datasets import load_dataset, DatasetDict


def load_ag_news(cfg, subset: bool = False) -> DatasetDict:
    dataset = load_dataset(cfg.name)
    if subset:
        train_size = cfg.train_subset_size
        test_size = cfg.test_subset_size
        dataset["train"] = dataset["train"].select(range(train_size))
        dataset["test"] = dataset["test"].select(range(test_size))
    return dataset
