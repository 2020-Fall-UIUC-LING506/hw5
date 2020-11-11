import torch

import argparse
import pathlib
from io import open
import unicodedata
import re


class Data:

    @staticmethod
    def clean(s: str) -> str:
        """Turn a Unicode string to plain ASCII, thanks to
           https://stackoverflow.com/a/518232/2809427"""
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize(s: str) -> str:
        """Lowercase, trim, and remove non-letter characters"""
        s = Data.clean(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def read_corpus(file: str) -> str:
        """Read the file and split into lines"""
        lines = open(file, encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        text_corpus = ["<s> " + Data.normalize(l) + " </s>" for l in lines]

        return text_corpus


class Vocabulary:
    """Vocabulary manages the conversion of word representations from strings to and from integers."""
    def __init__(self):
        self.word2index = {"<s>": 0, "</s>": 1}
        self.index2word = {0: "<s>", 1: "</s>"}
        self.n_words = 2  # Count SOS and EOS

    def getIndex(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        return self.word2index[word]


class RNN(torch.nn.Module):
    def __init__(self, vocab: Vocabulary, hidden_size: int):
        super(RNN, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab.n_words, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, vocab.n_words)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(torch.tensor([[input]])).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    @staticmethod
    def train_lm(*,
                 training_file: str,
                 hidden_size: int,
                 learning_rate: float,
                 training_epochs: int,
                 batch_size: int,
                 verbosity: int) -> "RNN":

        # Read corpus from disk
        text_corpus = Data.read_corpus(training_file)

        # Convert each word in the corpus to an integer
        vocab = Vocabulary()
        numbered_corpus = [[vocab.getIndex(word) for word in sentence.split(' ')] for sentence in text_corpus]

        # We will use negative log likelihood as the loss criterion
        criterion = torch.nn.NLLLoss()

        # Create an RNN. Note that at this point the parameters of the RNN have not been trained.
        rnn = RNN(vocab, hidden_size)

        # We will use stochastic gradient descent to train the parameters of the RNN
        optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        # An epoch denotes one training run through the entire training corpus
        for epoch in range(training_epochs):

            total_loss = rnn.batch_train(batch_size, numbered_corpus, optimizer, criterion) if batch_size > 1 else rnn.train(numbered_corpus, optimizer, criterion)

            # TODO: After implementing the validation_loss method, add code here to perform early stopping if appropriate.
            #       See §7.8.4 of Neural Machine Translation by Philipp Koehn

            if epoch % 10 == 0 or epoch+1 == training_epochs or verbosity > 0:
                print(f"Loss after epoch {epoch}:\t{total_loss / len(numbered_corpus)}")

        # RNN LM has now been trained
        return rnn

    def train(self, numbered_corpus, optimizer, criterion) -> float:
        """Perform training over one sentence at a time. Returns the total loss over the corpus.

           * We must zero out the optimizer gradient before processing each sentence:
             * optimzer.zero_grad()
           * We must keep track of the loss at the sentence level by summing the loss over every word in the sentence:
             * sentence_loss += word_loss
           * After processing each sentence, we must run the backward method on the sentence loss:
             * sentence_loss.backward()
           * We must perform an optimizer step after processing each sentence:
             * optimizer.step()
        """
        # Keep track of the total loss
        #   (as measured by the negative log likelihood loss criterion)
        #   over the entire corpus during this epoch
        total_loss = 0.0

        # Iterate over every sentence in the training corpus.
        #   Each sentence is represented as a list of integers
        #   (each integer represents one word).
        for sentence in numbered_corpus:

            optimizer.zero_grad()
            sentence_loss = 0
            hidden = self.initHidden()
            for i in range(len(sentence) - 1):
                output, hidden = self.forward(sentence[i], hidden)
                word_loss = criterion(output, torch.tensor([sentence[i + 1]]))
                sentence_loss += word_loss
            total_loss += sentence_loss.data.item()
            sentence_loss.backward()
            optimizer.step()

        return total_loss

    def generate(self) -> str:
        """Generates a random sequence of text using the trained RNN LM.

           See §7.8.4 of Neural Machine Translation by Philipp Koehn

           Hint: see https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#sampling-the-network
        """
        # TODO: Implement this method
        pass

    def validation_loss(self, numbered_validation_corpus, criterion) -> float:
        """Calculate loss on a validation set, processing one sentence at a time. Returns the total loss over the validation set.

           See §7.8.4 of Neural Machine Translation by Philipp Koehn
        """
        # TODO: Implement this method
        pass

    def batch_train(self, batch_size, numbered_corpus, optimizer, criterion) -> float:
        """Perform training over one batch of sentences at a time. Returns the total loss over the corpus.

           See §7.8.4 of Neural Machine Translation by Philipp Koehn

           Divide the corpus into batches. Each batch should contain batch_size sentences.

           * We must zero out the optimizer gradient before processing each batch:
             * optimizer.zero_grad()
           * We must keep track of the loss at the batch level by summing the loss over sentence in the batch:
             * batch_loss += sentence_loss
           * After processing each batch, we must run the backward method on the batch loss:
             * batch_loss.backward()
           * We must perform an optimizer step after processing each batch:
             * optimizer.step()
        """
        # TODO: Implement this method
        pass


def parse_argument_flags() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description='Train an RNN language model')
    parser.add_argument("-i", "--input",
                        type=str,
                        default=f"{pathlib.Path(__file__).parent.absolute()}/data/tiny.txt",
                        dest="input",
                        help="File containing sentences to train on")
    parser.add_argument("-s", "--hidden-size",
                        type=int,
                        default=256,
                        dest="hiddensize",
                        help="Number of units in hidden layer (default=256)")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=100,
                        dest="epochs",
                        help="Number of epochs to use in training (default=100)")
    parser.add_argument("-r", "--learning-rate",
                        type=float,
                        default=0.01,
                        dest="lr",
                        help="Learning rate during training (default=0.01)")
    parser.add_argument("-b", "--batch-size",
                        type=int,
                        default=1,
                        dest="batch",
                        help="Number of sentences to process at a time (default=1)")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=1,
                        dest="verbosity",
                        help="Verbosity level, 0-3 (default=1)")
    return parser.parse_args()


if __name__ == "__main__":

    flags = parse_argument_flags()

    rnn_lm: RNN = RNN.train_lm(training_file=flags.input,
                               hidden_size=flags.hiddensize,
                               learning_rate=flags.lr,
                               training_epochs=flags.epochs,
                               batch_size=flags.batch,
                               verbosity=flags.verbosity)

    # TODO: After implementing the validation_loss method, add appropriate flag(s) and argument(s) to train_lm so that you can apply early stopping based on a validation set.
