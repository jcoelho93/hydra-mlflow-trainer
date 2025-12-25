from transformers import AutoModelForSequenceClassification


def build_model(cfg):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.name,
        num_labels=cfg.num_labels,
    )
