# Named Entity Recognition with BERT, BiLSTM and CRF

In this example, we'll see how to implement a [**Token Classification**](https://sunglasses-ai.github.io/classy/docs/getting-started/no-code/tasks/#token-classification) model that:
- Reads data from a custom CoNLLu-like format, and
- Implements a noisification strategy that, with some probability, lowercases the whole input sentence, and
- Trains using a custom optimizer (Adamax - [:scroll: paper link](https://arxiv.org/abs/1412.6980)), and
- Uses a BERT-based model, with a BiLSTM and a CRF on top, and finally
- Evaluates using a Span-based F1 score.


## 📖 Example Explanation

### 🤓 [Customizing the data format](https://sunglasses-ai.github.io/classy/docs/getting-started/overriding-code/custom-data-form/)

In `classy`, by default, only two input formats are supported: [`jsonl`](https://sunglasses-ai.github.io/classy/docs/getting-started/no-code/input_formats/#jsonl) and [`tsv`](https://sunglasses-ai.github.io/classy/docs/getting-started/no-code/input_formats/#tsv). Our data, stored under `data/conll/en/{split}.conllu`, does not follow any of these two conventions and thus requires us either converting it to a `classy`-compatible format, or to write our own DataReader.

For this example, we'll go with the latter option. Following [the documentation](https://sunglasses-ai.github.io/classy/docs/getting-started/customizing-things/custom-data-format/), we implement a custom Data Reader that is able to read our custom format and convert it into a sequence of [`TokensSample`](). You can find the implementation [here](src/conllu_data_driver.py).

The only really important part is how to register your reader for `classy` to discover. For instance, look at the last line of the source file! That's all `classy` requires.

The neat part is that `classy` reads through your code automatically! No need to specify any `PYTHONPATH` or anything of the sorts :rocket:


### 📚 [Customizing the dataset](https://sunglasses-ai.github.io/classy/docs/advanced/custom-dataset/)

Let us assume that we want our model to be robust to capitalization features of the dataset. By that, we refer to the intrinsic biases that models pick up towards capitalized words when being trained on a Named Entity Recognition task. Now, think about how often you have seen the name of a city or a person without a capital letter. Should our model be robust to this kind of noise? Why not! Then we should not touch the data itself, but work on the dataset that converts data to batches that are later fed to the model.

`classy` works primarily with [IterableDatasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets), and thus implements a [function that returns an iterator of samples](src/noisy_ner_dataset.py#L89). Simply put, given some noise value, we [randomly lowercase all inputs](src/noisy_ner_dataset.py#L101) to the model, and yield the resulting batch so as to train on it.


### 🏋 [Customizing the optimizer](https://sunglasses-ai.github.io/classy/docs/advanced/custom-optimizer/)

For the sake of this example, we also implement a custom optimizer based on Adamax ([:scroll: paper link](https://arxiv.org/abs/1412.6980)), a variant of Adam that uses infinite norms and has been shown to perform very well on tasks where embeddings are rather important.

Again, this is quite simple. We just need to implement a `classy.optim.factories.Factory` which provides the right optimizer implementation once it is called. if you're interested, check out [the implementation](src/optim_adamax.py), it is only 30 lines long! (half of which are comments or imports :smile:)


### 🤖 [Customizing the model](https://sunglasses-ai.github.io/classy/docs/getting-started/overriding-code/custom-model/)

Finally, arguably the most important piece of the puzzle: our model!

For this example, we decided to augment our base Transformer model with a Bidirectional LSTM and a CRF layer on top, which is a great addition to your model in case you have to handle sequences (for example in NER, where you have entities that span across multiple tokens).

Very briefly, a [`ClassyPLModule`]() is a [`pl.LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) that also has a `predict_batch` function that is required for inference. Everything else is inherited from Pytorch Lightning's module and is needed to work with Pytorch Lightning. If you're a PL user, you probably know what you're doing. If you're not, check out [PL's amazing guides](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)!

In case you want some more in-depth details, check out [the code itself](src/model.py), it's well commented! 

##### CRF implementation

All credits for the CRF implementation go to [s14t284](https://github.com/s14t284), the original author of the package [TorchCRF](https://github.com/s14t284/TorchCRF/), which we have imported in our code (under `src/crf.py`) and slightly modified to work with `PyTorch 1.9.0`.


### 💯 [Customizing evaluation metric(s)](https://sunglasses-ai.github.io/classy/docs/advanced/custom-evaluation-metric/)

To conclude, we remind you that **evaluating your model properly** is just as important as training it. And, for the case of predicting labels that span across multiple tokens (which is exactly what NER does), it is extremely important to evaluate on a *span-based metric*. 

We do just that with our [`SpanF1PredictionCallback`](src/span_f1_callback.py), a `classy.pl_callbacks.PredictionCallback` that uses [`seqeval`](https://github.com/chakki-works/seqeval) (under [`datasets`](https://github.com/huggingface/datasets)) to evaluate the predictions emitted by our model.


## 🎁 Let's wrap things up!

We walked through all the main components of `classy`, but we have not mentioned how to actually train our newly-defined model!

`classy` is based on configuration files, which define virtually everything you might need to properly train your model. Specifically, each folder under `configurations` is mirrored from `classy`'s default configuration structure, and lets you add any new `.yaml` config file under the corresponding folder.

Additionally, we introduce a [custom profile](https://sunglasses-ai.github.io/classy/docs/getting-started/structured-configs/profiles/) under `configurations/profiles` that specifies all the needed configurations under their respective keys (e.g., data, model, callbacks), providing the initialization parameters for every object that will need to be instantiated for training.

Finally, we can launch our training by running `classy train token data/conll/en -n <exp-name> --profile ner-crf`.
