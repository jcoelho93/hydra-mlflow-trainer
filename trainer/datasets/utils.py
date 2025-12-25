from transformers import AutoTokenizer
from datasets import DatasetDict


def tokenize_dataset(cfg, dataset: DatasetDict) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    def preprocess(examples):
        return tokenizer(
            examples[cfg.dataset.text_field],
            truncation=True,
            padding="max_length",
            max_length=cfg.dataset.max_length,
        )

    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", cfg.dataset.label_field])
    return tokenized
