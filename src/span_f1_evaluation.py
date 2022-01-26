from typing import Dict, List

from datasets import load_metric

from classy.data.data_drivers import TokensSample
from classy.evaluation.base import Evaluation


class SpanF1Evaluation(Evaluation):
    """
    This class computes the span-f1, span-precision and span-recall metrics using the "seqeval" metrics from the
    HuggingFace datasets library.
    """

    def __init__(self):
        self.backend_metric = load_metric("seqeval")

    def __call__(self, path: str, predicted_samples: List[TokensSample]) -> Dict:
        """

        Args:
            path: Path to the dataset being evaluated
            predicted_samples: a list of TokensSamples containing the reference_annotations
             and the predicted annotations

        Returns:
            a dictionary containing the name of the metrics and their values

        """
        metric_out = self.backend_metric.compute(
            predictions=[sample.predicted_annotation for sample in predicted_samples],
            references=[sample.reference_annotation for sample in predicted_samples],
        )
        p, r, f1 = (
            metric_out["overall_precision"],
            metric_out["overall_recall"],
            metric_out["overall_f1"],
        )

        return {"span_precision": p, "span_recall": r, "span_f1": f1}
