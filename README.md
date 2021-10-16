# :dark_sunglasses: [Classy](https://github.com/sunglasses-ai/classy) Examples

Welcome! If you're landing here, it's probably because you want to try out our library, [classy](https://github.com/sunglasses-ai/classy).

This repository hosts all the examples that the `classy` team and the `classy` community have developed for you to use. 

Don't forget that all these examples have been built from [`classy`'s template](https://github.com/sunglasses-ai/classy-template)! Check it out to have more info on how to get started.

### How do I browse examples?

You can use the built-in branch selector at the top of this page :)

### How do I clone an example?

Super simple! Just run `git clone --depth 1 -b example-name --single-branch https://github.com/sunglasses-ai/classy-examples.git [target-folder-name]`. `[target-folder-name]` is optional, if it is not specified it will clone into `classy-examples` (we suggest cloning into `classy-<example-name>` or `classy-examples/<example-name>`).


## :heart: Contributing

If you want to contribute, first of all, that's great, thank you!

In general, contributing guidelines from `classy`'s main repository apply here as well. In particular, we want our examples to be as clear as possible, with commented code and **a great README**.

If you have some contribution to make, either to a specific example or to add a new example, open a pull request and we'll work through it together!


## :sos: Help / get in touch

Finally, if you need help, want to report a bug, or just want to get in touch, [open an issue](https://github.com/sunglasses-ai/classy/issues/new) on the main `classy` repository or join us on [Slack](https://sunglasses-ai.slack.com)!

## :rocket: Examples

This is a list of the available examples with a brief description and the `classy` components that they interact with.

Link | Description | Interactions
--- | --- | ---
[ner-crf](https://github.com/sunglasses-ai/classy-examples/tree/ner-crf) | A BERT-based architecture for Named Entity Recognition, with a BiLSTM and a CRF layer on top. | Custom Model, DataReader (custom CoNLLu format), Dataset (random lowercasing of input), Metric (Span F1) and Optimizer (Adamax)!
[consec](https://github.com/sunglasses-ai/classy-examples/tree/consec) | Implementation of the [ConSeC paper :scroll:](TODO) | **anything** (note: this is very advanced and might be overwhelming for many users. We recommend checking it out only if you need to do something very complex with `classy` or if you're looking for ConSeC's implementation.)
