# Named Entity Recognition with BERT, BiLSTM and CRF

In this example, we'll see how to implement a [**Token Classification**](https://sunglasses-ai.github.io/classy/docs/reference-manual/tasks-and-formats/#token-classification) model that:
- Reads data from a custom CoNLLu-like format, and
- Implements a noisification strategy that, with some probability, lowercases the whole input sentence, and
- Trains using a custom optimizer (Adamax - [:scroll: paper link](https://arxiv.org/abs/1412.6980)), and
- Uses a BERT-based model, with a BiLSTM and a CRF on top, and finally
- Evaluates using a Span-based F1 score.


## 📖 Example Explanation

### 🤓 [Customizing the data format](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-data-format/)

In `classy`, by default, only two input formats are supported: [`jsonl`](https://sunglasses-ai.github.io/classy/docs/reference-manual/tasks-and-formats/#jsonl-2) and [`tsv`](https://sunglasses-ai.github.io/classy/docs/reference-manual/tasks-and-formats/#tsv-2). Our data, stored under `data/conll/en/{split}.conllu`, does not follow any of these two conventions and thus requires us either converting it to a `classy`-compatible format, or to write our own DataReader.

For this example, we'll go with the latter option. Following [the documentation](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-data-format/), we implement a custom Data Reader that is able to read our custom format and convert it into a sequence of [`TokensSample`](https://sunglasses-ai.github.io/classy/docs/api/data/data_drivers/#TokensSample). You can find the implementation [here](src/conllu_data_driver.py).

The only really important part is how to register your reader for `classy` to discover. For instance, look at the last line of the source file! That's all `classy` requires.

The neat part is that `classy` reads through your code automatically! No need to specify any `PYTHONPATH` or anything of the sorts :rocket:


### 📚 [Customizing the dataset](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-dataset/)

Let us assume that we want our model to be robust to capitalization features of the dataset. By that, we refer to the intrinsic biases that models pick up towards capitalized words when being trained on a Named Entity Recognition task. Now, think about how often you have seen the name of a city or a person without a capital letter. Should our model be robust to this kind of noise? Why not! Then we should not touch the data itself, but work on the dataset that converts data to batches that are later fed to the model.

`classy` works primarily with [IterableDatasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets), and thus implements a [function that returns an iterator of samples](src/noisy_ner_dataset.py#L104). Simply put, given some noise value, we [randomly lowercase all inputs](src/noisy_ner_dataset.py#L118) to the model, and yield the resulting batch so as to train on it.


### 🏋 [Customizing the optimizer](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-optimizer/#custom-optimizers)

For the sake of this example, we also implement a custom optimizer based on Adamax ([:scroll: paper link](https://arxiv.org/abs/1412.6980)), a variant of Adam that uses infinite norms and has been shown to perform very well on tasks where embeddings are rather important.

Again, this is quite simple. We just need to implement a `classy.optim.factories.Factory` which provides the right optimizer implementation once it is called. if you're interested, check out [the implementation](src/optim_adamax.py), it is only 30 lines long! (half of which are comments or imports :smile:)


### 🤖 [Customizing the model](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-model/)

Finally, arguably the most important piece of the puzzle: our model!

For this example, we decided to augment our base Transformer model with a Bidirectional LSTM and a CRF layer on top, which is a great addition to your model in case you have to handle sequences (for example in NER, where you have entities that span across multiple tokens).

Very briefly, a [`ClassyPLModule`]() is a [`pl.LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) that also has a `predict_batch` function that is required for inference. Everything else is inherited from Pytorch Lightning's module and is needed to work with Pytorch Lightning. If you're a PL user, you probably know what you're doing. If you're not, check out [PL's amazing guides](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)!

In case you want some more in-depth details, check out [the code itself](src/model.py), it's well commented! 

##### CRF implementation

All credits for the CRF implementation go to [s14t284](https://github.com/s14t284), the original author of the package [TorchCRF](https://github.com/s14t284/TorchCRF/), which we have imported in our code (under `src/crf.py`) and slightly modified to work with `PyTorch 1.9.0`.


### 💯 [Customizing evaluation metric(s)](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-metric/)

To conclude, we remind you that **evaluating your model properly** is just as important as training it. 
And, for the case of predicting labels that span across multiple tokens (which is exactly what NER does), it is crucial to evaluate using a *span-based metric*. 

We do that with our [`SpanF1Evaluation`](src/span_f1_evaluation.py), a `classy.evaluation.Evaluation` class that uses [`seqeval`](https://github.com/chakki-works/seqeval) (under [`datasets`](https://github.com/huggingface/datasets)) to evaluate the predictions emitted by our model.

To log the metrics at training time you only have to add the following parameters to the train command.

```
-c callbacks=evaluation
```

`callbacks=evaluation` tells classy to insert a callback in the evaluation loop that leverages you evaluation class (`SpanF1Evaluation`) to compute the defined metrics. 

To use the logged metrics (e.g. _span_f1_) as the monitor values for both  model selection and early stopping you simply have to postpend the following parameter after the previous one:
```
callbacks_monitor=validation_span_f1
```
Please notice that _validation_ is prepended to _span_f1_ (that is the original metric name in the _SpanF1Evaluation_ class) because we are using the _callbacks=evaluation_ param that automatically adds it.

## 🎁 Let's wrap things up!

`classy` is based on configuration files, which define virtually everything you might need to properly train your model. Here, we introduced a [custom profile](https://sunglasses-ai.github.io/classy/docs/reference-manual/structured-configs/overall-structure/) under `configurations/profiles` that specifies all the parameters that we changed from the default ones. If you want to check the complete configuration there are two possible ways: 
1. adding the `--print` argument to the `train` command.
2. checking the config.yaml file in you experiment directory under _.hydra_.

We walked through all the main components of `classy`, but we have not mentioned how to actually train our newly-defined model!

### Training the Model

Finally, we can launch our training by running: 
```bash
classy train token data/conll/en -n <exp-name> --profile ner-crf -c callbacks=evaluation callbacks_monitor=validation_span_f1
```
