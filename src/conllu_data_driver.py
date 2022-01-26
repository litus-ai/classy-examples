from typing import Iterator, Generator

from classy.data.data_drivers import READERS_DICT, TOKEN, TokensDataDriver, TokensSample


class CoNLLUTokensDataDriver(TokensDataDriver):
    """
    This is a very simple class where you define
    1) how to convert a dataset in the CoNLLU format into a generator
    of ~classy.data.data_drivers.TokensSample objects and
    2) how to store a collection of TokensSample into a file
    written following the CoNLLU format.

    basically:
        file.conllu -> Generator[TokensSample]
        Generator[TokensSample] -> file.conllu

    """

    def read(self, lines: Iterator[str]) -> Generator[TokensSample, None, None]:
        """
        Method to convert a list of lines coming from a file in the CoNLLU format
        to a generator of ~classy.data.data_drivers.TokensSample.

        Args:
            lines: an Iterator over the lines of a file in input

        Returns:
            a generator over TokensSample
        """
        tokens = []
        labels = []

        for line in lines:

            if "DOCSTART" in line:
                continue

            if line.startswith("#"):
                continue

            if len(line) == 0:
                if len(tokens) > 0:
                    yield TokensSample(tokens, labels)
                tokens, labels = [], []
                continue

            _, token, label = line.split("\t")
            tokens.append(token)
            labels.append(label)

        if len(tokens) > 0:
            yield TokensSample(tokens, labels)

    def save(
        self,
        samples: Iterator[TokensSample],
        path: str,
        use_predicted_annotation: bool = False,
    ) -> None:
        """
        Method to store a collection of ~classy.data.data_drivers.TokensSample
         into a file following the CoNLLU format.

        Args:
            samples: an Iterator over a collection of ~classy.data.data_drivers.TokensSample
            path: the file path where the samples will be stored
            use_predicted_annotation: whether to use the predicted_annotation
                instead of the reference annotation

        Returns:
            None
        """

        with open(path, "w") as f:
            for sample in samples:
                labels = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else sample.reference_annotation
                )
                for i, (token, label) in enumerate(zip(sample.tokens, labels)):
                    f.write(f"{i}\t{token}\t{label}\n")
                f.write("\n")


# this is how you specify that when the task is a token-level classification and the data format is 'conllu' the
# dataset must use the CoNLLUTokensDataDriver to parse di samples.
READERS_DICT[TOKEN, "conllu"] = CoNLLUTokensDataDriver
