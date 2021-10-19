import logging
from typing import List, Tuple

import pytorch_lightning as pl

from classy.data.data_drivers import TokensSample, TOKEN
from classy.pl_callbacks.prediction import PredictionCallback
from classy.pl_modules.base import ClassyPLModule
from datasets import load_metric

logger = logging.getLogger(__name__)


class SpanF1PredictionCallback(PredictionCallback):
    """
    This class computes the span-f1, span-precision and span-recall metrics using the "seqeval" metrics from the
    HuggingFace datasets library.

    Inheriting from the PredictionCallback let's you define you own evaluation metrics, just by taking in input the
    samples and their relative predictions. All you have to do is implementing the __call__ method

    """
    def __init__(self):
        self.backend_metric = load_metric("seqeval")

    def __reset_trainer_metrics(self, callback_name: str, trainer: pl.Trainer) -> None:
        # todo lightning reset of metrics is yet unclear: fix once it becomes clear and delete the following block
        for metric in trainer._results.result_metrics:
            if metric.meta.name in [f"{callback_name}_precision", f"{callback_name}_recall", f"{callback_name}_f1"]:
                metric.reset()

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[TokensSample, List[str]]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        assert (
            model.task == TOKEN
            and isinstance(predicted_samples[0][0], TokensSample)
            and isinstance(predicted_samples[0][1], list)
        )

        # call to the seqeval metric that will compute the span-f1/precision/recall
        metric_out = self.backend_metric.compute(
            predictions=[labels for _, labels in predicted_samples],
            references=[sample.labels for sample, _ in predicted_samples],
        )

        p = metric_out["overall_precision"]
        r = metric_out["overall_recall"]
        f1 = metric_out["overall_f1"]

        # reset trainer metrics
        self.__reset_trainer_metrics(callback_name=name, trainer=trainer)

        # log new metrics metrics
        model.log(f"{name}_precision", p, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_recall", r, prog_bar=True, on_step=False, on_epoch=True)
        model.log(f"{name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        logger.info(
            f"SpanF1PredictionCallback with name {name} completed with scores (p={p:.4f}, r={r:.4f}, f1={f1:.4f})"
        )
