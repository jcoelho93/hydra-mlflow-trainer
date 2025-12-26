import os
import torch
from torch.utils.data import DataLoader
import mlflow
from datasets import DatasetDict
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
    def __init__(self, model, dataset: DatasetDict, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.train_loader = DataLoader(dataset["train"], batch_size=cfg.trainer.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset["test"], batch_size=cfg.trainer.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.model.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.checkpoint_dir = cfg.trainer.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_epoch = 0
        self._load_checkpoint_if_exists()

    def _checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, "latest.pt")

    def _save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, self._checkpoint_path())
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint_if_exists(self):
        path = self._checkpoint_path()
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from checkpoint at epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.trainer.epochs):
            self.model.train()
            all_preds, all_labels = [], []
            total_loss = 0

            batches = enumerate(self.train_loader)
            print(f"Processing {len(self.train_loader)} batches for epoch {epoch + 1}")
            for batch_idx, batch in batches:
                print(f"Processing batch {batch_idx + 1}/{len(self.train_loader)}")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch[self.cfg.dataset.label_field].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                if batch_idx % self.cfg.trainer.log_every == 0:
                    print(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                    step = epoch * len(self.train_loader) + batch_idx
                    mlflow.log_metric("batch_loss", loss.item(), step=step)

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted")
            print(f"Epoch {epoch} | Loss {total_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")

            epoch_loss = total_loss / len(self.train_loader)
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)

            self._save_checkpoint(epoch)
