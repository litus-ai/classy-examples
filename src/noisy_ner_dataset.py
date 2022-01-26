from typing import Iterator, Iterable, Dict, Any, Optional, Tuple, List, Callable, Union

import numpy as np
import torch

from classy.data.data_drivers import TokensSample
from classy.data.dataset.base import BaseDataset, batchify
from classy.utils.vocabulary import Vocabulary


import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class NoisyNERDataset(BaseDataset):
    """
    This is a very simple dataset that applies a noisification to the dataset by randomly lowercasing all the tokens
    in an input sentence.

    The dataset you declare must implement ~classy.data.dataset.base.BaseDataset, in this way you just have to declare
        1) how your sample have to be converted to batch elements in the dataset_iterator_func method,
        2) how you will batch these elements in the _init_fields_batcher method and
        3) how you can fit a vocabulary starting a collection of TokensSample

    To have a more in depth overview of how to implement a custom dataset please refer to the documentation under
    Advanced / Custom Dataset

    """

    @staticmethod
    def fit_vocabulary(samples: Iterator[TokensSample]) -> Vocabulary:
        """
        This method computes a Vocabulary containing an index with all the possible classes in your dataset.
        Args:
            samples: a collection of samples from which we can compute the possible classes

        Returns:
            the Vocabulary containing all the possible classes in your dataset
        """
        return Vocabulary.from_samples(
            [
                {"labels": label}
                for sample in samples
                for label in sample.reference_annotation
            ]
        )

    def __init__(
        self,
        noisification_probability: float,
        transformer_model: str,
        samples_iterator: Callable[[], Iterator[TokensSample]],
        vocabulary: Vocabulary,
        tokens_per_batch: int,
        max_batch_size: int,
        section_size: int,
        prebatch: bool,
        materialize: bool,
        min_length: int,
        max_length: int,
        for_inference: bool,
    ):
        self.noisification_probability = noisification_probability
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

        super().__init__(
            samples_iterator=samples_iterator,
            vocabulary=vocabulary,
            batching_fields=["input_ids"],
            tokens_per_batch=tokens_per_batch,
            max_batch_size=max_batch_size,
            fields_batchers=None,
            section_size=section_size,
            prebatch=prebatch,
            materialize=materialize,
            min_length=min_length,
            max_length=max_length
            if max_length != -1
            else self.tokenizer.model_max_length,
            for_inference=for_inference,
        )

        self._init_fields_batcher()

    def _init_fields_batcher(self) -> None:
        """
        A method for declaring the you batch elements have to be aggregated in order to be processed by your model
        """
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "labels": lambda lst: batchify(
                lst, padding_value=-100
            ),  # -100 == cross entropy ignore index
            "samples": None,
            "token_offsets": None,
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:
        """
        In this method you declare how to transform a TokensSample in to an element that your model can process.
        Implementing BaseDataset you can assume to have access to a samples_iterator that let's you iterate over
        all the TokensSample in the dataset.

        Returns:
            an iterable over all the transformed elements in the dataset
        """
        for token_sample in self.samples_iterator():

            noisify_tokens = np.random.uniform(0, 1) < self.noisification_probability

            if noisify_tokens:
                input_ids, token_offsets = self._tokenize(
                    [token.lower() for token in token_sample.tokens]
                )
            else:
                input_ids, token_offsets = self._tokenize(token_sample.tokens)

            elem_dict = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "token_offsets": token_offsets,
            }
            if token_sample.reference_annotation is not None:
                elem_dict["labels"] = torch.tensor(
                    [
                        self.vocabulary.get_idx(
                            k="labels", elem=token_sample.reference_annotation[idx]
                        )
                        for idx in token_sample.target
                    ]
                )

            elem_dict["samples"] = token_sample
            yield elem_dict

    def _tokenize(
        self, tokens: List[str]
    ) -> Optional[Tuple[torch.Tensor, List[Tuple[int, int]]]]:
        tok_encoding = self.tokenizer.encode_plus(
            tokens, return_tensors="pt", is_split_into_words=True
        )
        try:
            return tok_encoding.input_ids.squeeze(0), [
                tuple(tok_encoding.word_to_tokens(wi)) for wi in range(len(tokens))
            ]
        except TypeError:
            logger.warning(f"Tokenization failed for tokens: {' | '.join(tokens)}")
            return None
