import torch
from nltk import word_tokenize
from collections import OrderedDict, Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        data,
        metadata,
        labels,
        max_length,
        vocab=None,
        min_freq=5,
    ):
        self.data = data
        # Set the maximum length we will keep for the sequences
        self.max_length = max_length

        # We then need to tokenize the data ..
        tokenized_data = [word_tokenize(seq) for seq in self.data]

        # Allow to import a vocabulary (for valid/test datasets, that will use the training vocabulary)
        if vocab is not None:
            self.word2idx, self.idx2word = vocab
        else:
            # If no vocabulary imported, build it (and reverse)
            self.word2idx, self.idx2word = self.build_vocab(tokenized_data, min_freq)

        # Transform words into lists of indexes
        indexed_data = [
            list(
                map(
                    lambda x: float(self.word2idx[x])
                    if (x in self.word2idx.keys())
                    else self.word2idx.get("UNK"),
                    tokenized_data[i],
                )
            )
            for i in range(len(tokenized_data))
        ]
        # And transform this list of lists into a list of Pytorch LongTensors
        tensor_data = [torch.LongTensor(seq) for seq in indexed_data]

        # And the labels into a FloatTensor
        tensor_y = torch.FloatTensor(labels)

        # To finally cut it when it's above the maximum length
        cut_tensor_data = [seq[: self.max_length] for seq in tensor_data]

        # Now, we need to use the pad_sequence function to have the whole dataset represented as one tensor,
        # containing sequences of the same length. We choose the padding_value to be 0, the we want the
        # batch dimension to be the first dimension
        self.tensor_data = pad_sequence(
            cut_tensor_data, batch_first=True, padding_value=0
        )
        self.tensor_y = tensor_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The iterator just gets one particular example with its category
        # The dataloader will take care of the shuffling and batching
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tensor_data[idx], self.tensor_y[idx]

    def vocabulary(corpus, voc_threshold=0):
        """
        Function using word counts to build a vocabulary - can be improved with a second parameter for
        setting a frequency threshold
        Params:
            corpus (list of list of strings): corpus of sentences
            voc_threshold (int): maximum size of the vocabulary (0 means no limit !)
        Returns:
            vocabulary (dictionary): keys: list of distinct words across the corpus
                                    values: indexes corresponding to each word sorted by frequency
        """

        # Setting limits
        voc_threshold = voc_threshold if voc_threshold else int(10e5)

        # Count each words of our corpus
        word_counts = dict(Counter(word_tokenize(" ".join(corpus))))

        # Sort them decreasingly
        sorted_word_counts = OrderedDict(
            sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        )

        # Create our vocabulary (no more than the parameter voc_threshold)
        vocabulary_word_counts = dict(list(sorted_word_counts.items())[:voc_threshold])

        vocabulary = {
            list(vocabulary_word_counts.keys())[i]: i
            for i in range(len(vocabulary_word_counts))
        }

        # Add 'Unknown' word
        vocabulary = {**vocabulary, "UNK": len(vocabulary_word_counts)}
        vocabulary_word_counts = {**vocabulary_word_counts, "UNK": 0}

        return vocabulary, vocabulary_word_counts

    def build_vocab(self, tokenized_data, count_threshold):
        """
        Same as in the previous TP: we want to output word_index, a dictionary containing words
        and their corresponding indexes as {word : indexes}
        But we also want the reverse, which is a dictionary {indexes: word}
        Don't forget to add a UNK token that we need when encountering unknown words
        We also choose '0' to represent the padding index, so begin the vocabulary index at 1 !
        """

        # Tokenize/Flatten our corpus
        tokenized_data_flatten = [
            item for sublist in tokenized_data for item in sublist
        ]

        # Count each words
        word_counts = Counter(tokenized_data_flatten)

        # Filtered only the ones above min frequency
        filtered_word_counts = {
            word: count
            for word, count in word_counts.items()
            if count >= count_threshold
        }

        # sort them by frequency
        word_counts = OrderedDict(
            sorted(filtered_word_counts.items(), key=lambda kv: kv[1], reverse=True)
        )

        # Add the index
        word_indexed = dict(
            zip(filtered_word_counts.keys(), range(1, len(filtered_word_counts) + 1))
        )

        # Add 'Unknown' word and 'padding' + reverse operation
        word_index = {"PAD": 0, **word_indexed, "UNK": len(word_indexed) + 1}
        idx_word = {v: k for k, v in word_index.items()}

        return word_index, idx_word

    def get_vocab(self):
        # A simple way to get the training vocab when building the valid/test
        return self.word2idx, self.idx2word
