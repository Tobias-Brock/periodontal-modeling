import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..training import Trainer, final_metrics
from ._basegraph import BaseGraphTrainer

try:
    from torch_geometric.loader import DataLoader
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The graph submodule requires 'torch_geometric'. Install the optional "
        "dependencies with 'pip install periomod[gnn]'."
    ) from error


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolves the compute device of the graph submodule.

    Args:
        device (Optional[str]): Requested device. Choose 'auto' to select CUDA
            when available, or any device accepted by PyTorch. Defaults to None,
            which is treated as 'auto'.

    Returns:
        torch.device: Device used for training and inference.
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GraphTrainer(BaseGraphTrainer):
    """Trainer for heterogeneous GNNs on batches of complete patient graphs.

    Training minimizes the loss over the target sites of each patient graph,
    while all remaining sites contribute as context through message passing.
    The validation loss or criterion drives early stopping and model selection,
    so that the test set remains untouched during model development.

    Evaluation reuses the criteria, threshold tuning and metrics of the tabular
    package, which makes the results directly comparable to the site level
    models of `periomod.benchmarking`.

    Inherits:
        - `BaseGraphTrainer`: Validates classification and criterion.

    Args:
        classification (str): Type of classification, either 'binary' or
            'multiclass'.
        criterion (str): Evaluation criterion for model selection (e.g., 'f1',
            'macro_f1', 'brier_score').
        lr (Optional[float]): Learning rate. Defaults to the configured value.
        weight_decay (Optional[float]): Weight decay of the optimizer. Defaults
            to the configured value.
        epochs (Optional[int]): Maximum number of training epochs. Defaults to
            the configured value.
        patience (Optional[int]): Number of epochs without improvement before
            early stopping. Defaults to the configured value.
        pos_weight (Optional[float]): Weight of the positive class in binary
            classification. Defaults to None.
        threshold_tuning (bool): Tunes the decision threshold on the validation
            sites when the criterion is 'f1', 'recall' or 'specificity'.
            Defaults to True.
        device (Optional[str]): Compute device. Defaults to the configured
            value.
        verbose (bool): Prints the training progress. Defaults to True.

    Attributes:
        lr (float): Learning rate of the optimizer.
        weight_decay (float): Weight decay of the optimizer.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs without improvement before stopping.
        pos_weight (Optional[float]): Weight of the positive class.
        threshold_tuning (bool): Indicates whether thresholds are tuned.
        verbose (bool): Controls verbosity of the training process.
        device (torch.device): Device used for training and inference.

    Methods:
        train: Trains a model with early stopping on the validation loader.
        predict: Computes site level predictions for a data loader.
        evaluate_model: Computes the final metrics of a data loader.

    Example:
        ```
        from periomod.graph import GraphTrainer

        trainer = GraphTrainer(classification="binary", criterion="f1")
        score, model, threshold = trainer.train(
            model=model, train_loader=train_loader, val_loader=val_loader
        )
        metrics = trainer.evaluate_model(
            model=model, loader=val_loader, threshold=threshold
        )
        ```
    """

    def __init__(
        self,
        classification: str,
        criterion: str,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        pos_weight: Optional[float] = None,
        threshold_tuning: bool = True,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the trainer with optimization and evaluation settings."""
        super().__init__(classification=classification, criterion=criterion)
        self.lr = self.lr if lr is None else lr
        self.weight_decay = self.weight_decay if weight_decay is None else weight_decay
        self.epochs = self.epochs if epochs is None else epochs
        self.patience = self.patience if patience is None else patience
        self.pos_weight = pos_weight
        self.threshold_tuning = threshold_tuning
        self.verbose = verbose
        self.device = resolve_device(device=self.device if device is None else device)
        self._evaluator = Trainer(
            classification=self.classification,
            criterion=self.criterion,
            tuning=None,
            hpo=None,
        )

    @property
    def maximize(self) -> bool:
        """Indicates whether the criterion is maximized.

        Returns:
            bool: False for 'brier_score', which is minimized, True otherwise.
        """
        return self.criterion != "brier_score"

    def _loss_fn(self) -> nn.Module:
        """Creates the loss function matching the classification type.

        Returns:
            nn.Module: Loss function applied to the target sites.
        """
        if self.classification == "binary":
            pos_weight = (
                None
                if self.pos_weight is None
                else torch.tensor([self.pos_weight], device=self.device)
            )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.CrossEntropyLoss()

    def _batch_loss(self, model: nn.Module, batch: Any, loss_fn: nn.Module):
        """Computes the loss over the target sites of a batch.

        Args:
            model (nn.Module): Heterogeneous GNN.
            batch (Any): Batch of patient graphs.
            loss_fn (nn.Module): Loss function applied to the target sites.

        Returns:
            Optional[torch.Tensor]: Loss of the batch, or None if the batch
                contains no target sites.
        """
        logits = model(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict)
        mask = batch["site"].target_mask
        if not bool(mask.any()):
            return None

        target = batch["site"].y[mask]
        if self.classification == "binary":
            return loss_fn(logits[mask].squeeze(-1), target.float())
        return loss_fn(logits[mask], target.long())

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Runs a single training or evaluation epoch.

        Args:
            model (nn.Module): Heterogeneous GNN.
            loader (DataLoader): Loader of patient graphs.
            loss_fn (nn.Module): Loss function applied to the target sites.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer of the
                training epoch. Defaults to None, which evaluates the loader.

        Returns:
            float: Mean loss over the batches containing target sites.
        """
        model.train(mode=optimizer is not None)
        losses = []

        for batch in loader:
            batch = batch.to(self.device)
            with torch.set_grad_enabled(mode=optimizer is not None):
                loss = self._batch_loss(model=model, batch=batch, loss_fn=loss_fn)
            if loss is None:
                continue
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else float("nan")

    def predict(
        self, model: nn.Module, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Computes site level predictions for a data loader.

        Args:
            model (nn.Module): Trained heterogeneous GNN.
            loader (DataLoader): Loader of patient graphs.

        Returns:
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]: Outcomes of the target
                sites, their predicted probabilities and a DataFrame identifying
                every target site by patient, tooth and side.
        """
        model.eval()
        outcomes, probabilities, identifiers = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = model(
                    x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict
                )
                mask = batch["site"].target_mask
                if not bool(mask.any()):
                    continue
                if self.classification == "binary":
                    probs = torch.sigmoid(logits[mask].squeeze(-1))
                else:
                    probs = torch.softmax(logits[mask], dim=-1)
                outcomes.append(batch["site"].y[mask].cpu().numpy())
                probabilities.append(probs.cpu().numpy())
                identifiers.append(
                    pd.DataFrame({
                        self.group_col: batch["site"].patient_id[mask].cpu().numpy(),
                        "tooth": batch["site"].tooth_id[mask].cpu().numpy(),
                        "side": batch["site"].side_id[mask].cpu().numpy(),
                    })
                )

        return (
            np.concatenate(outcomes),
            np.concatenate(probabilities),
            pd.concat(identifiers, ignore_index=True),
        )

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[float, nn.Module, Optional[float]]:
        """Trains a model with early stopping on the validation loader.

        The parameters of the epoch with the best validation criterion are
        restored after training.

        Args:
            model (nn.Module): Heterogeneous GNN to train.
            train_loader (DataLoader): Loader of the training patient graphs.
            val_loader (DataLoader): Loader of the validation patient graphs.

        Returns:
            Tuple[float, nn.Module, Optional[float]]: Best validation score,
                trained model and the corresponding decision threshold.
        """
        model = model.to(self.device)
        loss_fn = self._loss_fn()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_score = -np.inf if self.maximize else np.inf
        best_threshold: Optional[float] = None
        best_state = copy.deepcopy(model.state_dict())
        best_epoch, epochs_without_improvement = 0, 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._epoch(
                model=model, loader=train_loader, loss_fn=loss_fn, optimizer=optimizer
            )
            val_loss = self._epoch(model=model, loader=val_loader, loss_fn=loss_fn)
            outcomes, probs, _ = self.predict(model=model, loader=val_loader)
            score, threshold = self._evaluator.evaluate(
                y=outcomes, probs=probs, threshold=self.threshold_tuning
            )

            improved = score > best_score if self.maximize else score < best_score
            if improved:
                best_score, best_threshold, best_epoch = score, threshold, epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(
                    f"Epoch {epoch}: train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}, val {self.criterion} {score:.4f}"
                )

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping after {epoch} epochs.")
                break

        model.load_state_dict(best_state)
        if self.verbose:
            print(f"Best val {self.criterion}: {best_score:.4f} at epoch {best_epoch}.")
        return best_score, model, best_threshold

    def evaluate_model(
        self,
        model: nn.Module,
        loader: DataLoader,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes the final metrics of a data loader.

        Args:
            model (nn.Module): Trained heterogeneous GNN.
            loader (DataLoader): Loader of patient graphs.
            threshold (Optional[float]): Decision threshold of binary
                classification. Defaults to None, which applies 0.5.

        Returns:
            Dict[str, Any]: Evaluation metrics of the target sites.
        """
        outcomes, probs, _ = self.predict(model=model, loader=loader)
        if self.classification == "binary":
            preds = (probs >= (threshold if threshold is not None else 0.5)).astype(int)
        else:
            preds = np.argmax(probs, axis=1)

        return final_metrics(
            classification=self.classification,
            y=outcomes,
            preds=preds,
            probs=probs,
            threshold=threshold,
        )
