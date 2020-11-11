import torch

# wget http://gutenberg.net.au/ebooks01/0100021.txt
# perl -ne 's/\r//; print $_;' < 0100021.txt | /Users/pkoehn/Code/moses/scripts/ems/support/split-sentences.perl | grep -v '^<' | /Users/pkoehn/Code/moses/scripts/tokenizer/tokenizer.perl | more

from io import open
import unicodedata
import string
import re
import random

class Vocabulary:
    def __init__(self):
        self.word2index = {"<s>": 0, "</s>": 1}
        self.index2word = {0: "<s>", 1: "</s>"}
        self.n_words = 2  # Count SOS and EOS

    def getIndex(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        return self.word2index[word]

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readCorpus(file):
    # Read the file and split into lines
    lines = open(file, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    text_corpus = ["<s> " + normalizeString(l) + " </s>" for l in lines]

    return text_corpus

class RNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(torch.tensor([[input]])).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

text_corpus = readCorpus("rnn.py")
vocab = Vocabulary()
numbered_corpus = [[vocab.getIndex(word) for word in sentence.split(' ')] for sentence in text_corpus]

criterion = torch.nn.NLLLoss()

hidden_size = 256
learning_rate = 0.01

rnn = RNN(vocab.n_words, hidden_size)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

for epoch in range(100):
    total_loss = 0
    for sentence in numbered_corpus:
        #print(sentence)
        optimizer.zero_grad()
        sentence_loss = 0
        hidden = rnn.initHidden()
        for i in range(len(sentence)-1):
            output, hidden = rnn.forward( sentence[i], hidden )
            word_loss = criterion( output, torch.tensor([sentence[i+1]]))
            print("correct: %15s, loss: %5.2f, prediction: %s" % (vocab.index2word[sentence[i+1]], word_loss, vocab.index2word[torch.argmax(output, dim=1).item()]))
            sentence_loss += word_loss
        #print(loss.data.item())
        total_loss += sentence_loss.data.item();
        sentence_loss.backward()
        optimizer.step()
    print(total_loss / len(numbered_corpus))
