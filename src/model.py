from typing import Dict, Any, List, Iterator, Tuple, Optional

import omegaconf
import torch
import torch.nn as nn
import torchmetrics
from classy.data.data_drivers import TokensSample
from classy.data.dataset.base import batchify

from src.crf import CRF

from classy.pl_modules.base import ClassyPLModule

from classy.pl_modules.mixins.task import TokensTask
from classy.utils.vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, AutoModel


class CRFPLModule(TokensTask, ClassyPLModule):
    """
    This class describes the architecture and the behavior of our model. The architecture can be
    summarized as:
                          Transformer-based Encoder => BiLSTM => CRF

    To simplify classy and the organization of its code we use as
    a backbone PyTorch Lightning. For you this only means that
    your models must inherits from the ClassyPLModule and implement
    some methods that describe their behavior.

    All you need to know is that in all your modules you will have to implement the following 5 methods:
        1) forward: that describe a forward_pass of your model. Usually this is the method called from all the others
                    and is usually able of computing the output logits, the output predictions and the output loss.
        2) training_step: uses the forward method to compute and return the loss.
        3) validation_step: uses the forward method to compute the validation_loss and log metrics about the validation
                            set predictions.
        4) test_step: uses the forward method to compute  and log metrics about the test set predictions.
        5) batch_predict: receives in input all the parameters required for a forward pass and returns a list of
                          TokenSample that

    There is an example of all this methods implementation in this class.
    Go check the documentation for further details: Getting Started / Adapting the Code / Custom Model

    While it is not necessary for you to know the PyTorch Lightning framework in order to understand
    how this class is structured since PyTorch Lightning is just a way of organizing your code, it
    may clarify some doubts about its implementation.

    Here it is the link: https://www.pytorchlightning.ai (just in case)
    """

    def __init__(
        self,
        transformer_model: str,
        use_last_n_layers: int,
        fine_tune: bool,
        lstm_layers: int,
        lstm_hidden_size: int,
        lstm_bidirectional: bool,
        vocabulary: Vocabulary,
        optim_conf: omegaconf.DictConfig,
        dropout: float = 0.5,
    ):
        super().__init__(vocabulary=vocabulary, optim_conf=optim_conf)

        # store all the parameters in the __init__ function.
        # Needed for when you will load the model back
        self.save_hyperparameters(ignore="vocabulary")

        # Architecture Definition

        # Transformer-based encoder
        self.use_last_n_layers = use_last_n_layers
        self.transformer: PreTrainedModel = AutoModel.from_pretrained(
            transformer_model, output_hidden_states=True, return_dict=True
        )

        # If we are not fine tuning the transformer model then, we can
        # put it in eval mode and don't keep track of the gradients
        if not fine_tune:
            self.transformer.eval()
            self.transformer.requires_grad_(requires_grad=False)

        # Here we declare our BiLSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
        )

        num_classes = vocabulary.get_size("labels")
        self.pad_label_idx = vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)

        # And a very simple classification head
        self.classification_head = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size * (2 if lstm_bidirectional else 1), num_classes),
            nn.ReLU(),
        )

        # Finally we can declare our CRF module
        self.crf = CRF(num_classes, pad_idx=self.pad_label_idx)

        # let's define some standard metrics for classification tasks
        # here we use the torchmetrics library that contains useful
        # classes for common metrics such as Accuracy, Precision, ...
        ignore_index = vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)
        self.accuracy_metric = torchmetrics.Accuracy(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.p_metric = torchmetrics.Precision(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.r_metric = torchmetrics.Recall(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.micro_f1_metric = torchmetrics.F1(
            mdmc_average="global",
            ignore_index=ignore_index,
        )
        self.macro_f1_metric = torchmetrics.F1(
            num_classes,
            mdmc_average="global",
            average="macro",
            ignore_index=ignore_index,
        )

    def compute_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Simple utils to compute the loss via the crf module.
        :param logits: input logits (batch_size x seq_len x num_classes)
        :param labels: reference labels (batch_size x seq_len)
        :return:
        """
        crf_fwd = self.crf(logits, labels, mask=labels != self.pad_label_idx)
        loss: torch.Tensor = -(torch.sum(crf_fwd) / len(logits))
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_offsets: List[List[Tuple[int, int]]],
        samples: List[TokensSample],
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        compute_predictions: bool = True,
        compute_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output logits, predictions and loss.

        :param input_ids: (batch_size x seq_len) subwords ids computed by a Transformer Tokenizer
        :param attention_mask: (batch_size x seq_len) attention mask computed by a Transformer Tokenizer
        :param token_offsets: list of tokens offsets containing the corresponding subwords in which they have been split
        :param samples: a list containing the TokensSamples being classified
        :param token_type_ids: (batch_size x seq_len) contains the token_type_id of each token in the input_ids
        :param labels: (batch_size x seq_len)
        :param compute_predictions: whether or not to compute the predictions using the viterbi algorithm
        :param compute_loss: whether or not to compute the output loss
        :return: Dict containing the output logits, the predictions and the loss
        """
        word_representations = self.encode_words(input_ids, attention_mask, token_type_ids, token_offsets)
        logits: torch.FloatTensor = self.classification_head(word_representations)

        output_bag = {"logits": logits}

        if compute_predictions:
            tokens_mask = pad_sequence(
                [torch.ones(n) for n in map(len, token_offsets)], padding_value=0, batch_first=True
            ).bool()
            predictions: List[List[int]] = self.crf.viterbi_decode(logits, mask=tokens_mask)
            predictions: List[torch.Tensor] = [torch.tensor(seq, device=self.device) for seq in predictions]
            predictions: torch.Tensor = batchify(predictions, self.pad_label_idx)
            output_bag["predictions"] = predictions

        if compute_loss and labels is not None:
            labels[labels == -100] = self.pad_label_idx
            output_bag["loss"] = self.compute_loss(logits, labels)

        return output_bag

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        This method defines the training_step, where you compute the training loss.
        :param batch: batch in input with all the fields required for the forward method
        :param batch_idx: the index of the batch you are processing
        :return: the loss
        """
        classification_output = self.forward(**batch, compute_predictions=False, compute_loss=True)
        self.log("train_loss", classification_output["loss"])
        return classification_output["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        This method defines the validation_step, where you compute
        the validation loss and update the validation metrics.
        :param batch: batch in input with all the fields required for the forward method
        :param batch_idx: the index of the batch you are processing
        :return: None
        """
        classification_output = self.forward(**batch, compute_predictions=True, compute_loss=True)
        loss = classification_output["loss"]
        predictions = classification_output["predictions"]

        labels = batch["labels"].clone()
        labels[labels == -100] = self.vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)

        self.accuracy_metric(predictions, labels)
        self.p_metric(predictions, labels)
        self.r_metric(predictions, labels)
        self.micro_f1_metric(predictions, labels)
        self.macro_f1_metric(predictions, labels)

        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy_metric, prog_bar=True)
        self.log("val_precision", self.p_metric)
        self.log("val_recall", self.r_metric)
        self.log("val_micro-f1-score", self.micro_f1_metric, prog_bar=True)
        self.log("val_macro-f1-score", self.macro_f1_metric, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        This method defines the test_step, where you compute and update the validation metrics.
        :param batch: batch in input with all the fields required for the forward method
        :param batch_idx: the index of the batch you are processing
        :return: None
        """
        predictions = self.forward(**batch, compute_loss=False)["predictions"]

        labels = batch["labels"].clone()
        labels[labels == -100] = self.vocabulary.get_idx(k="labels", elem=Vocabulary.PAD)

        self.accuracy_metric(predictions, labels)
        self.p_metric(predictions, labels)
        self.r_metric(predictions, labels)
        self.micro_f1_metric(predictions, labels)
        self.macro_f1_metric(predictions, labels)

        self.log("test_accuracy", self.accuracy_metric)
        self.log("test_precision", self.p_metric)
        self.log("test_recall", self.r_metric)
        self.log("test_micro-f1-score", self.micro_f1_metric)
        self.log("test_macro-f1-score", self.macro_f1_metric)

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[TokensSample, List[str]]]:
        """
        Receives in input all the parameters required for a forward pass and returns a list containing tuples
        with each input TokenSample along with its tokens predictions.
        """
        samples = kwargs.get("samples")
        predictions = self.forward(*args, **kwargs, compute_predictions=True, compute_loss=False)["predictions"]
        for sample, prediction in zip(samples, predictions):
            yield sample, [
                self.vocabulary.get_elem(k="labels", idx=_p.item()) for _p in prediction[: len(sample.tokens)]
            ]

    def encode_words(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        token_offsets: List[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        """
        Simple abstraction to encode the input_ids into words representations using both the transformer_based encoder
        and the BiLSTM.
        """
        transformer_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            transformer_input["token_type_ids"] = token_type_ids
        encoder_out = self.transformer(**transformer_input)

        # encode bpes and merge them (sum strategy)
        if self.hparams.use_last_n_layers > 1:
            encoded_bpes = torch.stack(
                encoder_out[2][-self.hparams.use_last_n_layers :],
                dim=-1,
            ).sum(-1)
        else:
            encoded_bpes = encoder_out[0]

        # bpe -> token (first strategy)
        encoded_tokens = torch.zeros(
            (input_ids.shape[0], max(map(len, token_offsets)), encoded_bpes.shape[-1]),
            dtype=encoded_bpes.dtype,
            device=encoded_bpes.device,
        )

        for i, sample_offsets in enumerate(token_offsets):
            encoded_tokens[i, : len(sample_offsets)] = torch.stack([encoded_bpes[i, sj] for sj, ej in sample_offsets])

        encoded_tokens = self.lstm(encoded_tokens)[0]

        return encoded_tokens

    def on_train_epoch_start(self) -> None:
        """
        This is a simple hook to ensure that at the start of the training
        epoch the transformer model is still in the eval state.
        :return: None
        """
        if not self.hparams.fine_tune:
            self.transformer.eval()
