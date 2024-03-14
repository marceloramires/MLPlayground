import string
from io import open
import glob
import os
import unicodedata
import torch.nn as nn
import torch
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Parameters to load datasets
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Training parameters
n_hidden = 128
n_iters = 100000
print_every = 5000
plot_every = 1000
criterion = nn.NLLLoss()
learning_rate = 0.005

# Plotting parameters
n_confusion = 10000

class SurnameNationalityClassifyingModel:
    def __init__(self):
        self.category_lines = {}
        self.all_categories = []

    def load_saved_model(self, state_path):
        self.rnn = RNN(n_letters, n_hidden, self.n_categories)
        self.rnn.load_state_dict(torch.load(state_path))
        self.rnn.eval()

    def save_model(self, state_path):
        torch.save(self.rnn.state_dict(), state_path)

    def load_categories(self, dataset_path):
        for filename in self._find_files(dataset_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
        self.n_categories = len(self.all_categories)

    def get_categories(self):
        return self.all_categories
    
    def load_dataset(self, dataset_path):
        for filename in self._find_files(dataset_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self._read_lines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)

    def predict(self, name, n_predictions=3):
        with torch.no_grad():
            output = self._evaluate(self._line_to_tensor(name))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                predictions.append([value, self.all_categories[category_index]])
        return predictions

    def train(self):
        self.rnn = RNN(n_letters, n_hidden, self.n_categories)
        current_loss = 0
        all_losses = []

        def time_since(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

        start = time.time()

        for iter in range(1, n_iters + 1):
            category, line, category_tensor, line_tensor = self._random_training_example()
            output, loss = self._train_line(category_tensor, line_tensor)
            current_loss += loss

            # Print ``iter`` number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = self._category_from_output(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

        self._plot_training_losses(all_losses)

    def _evaluate(self, line_tensor):
        hidden = self.rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)

        return output

    def _find_files(self, path): 
        return glob.glob(path)
    
    def _unicode_to_ascii(self, string):
        return ''.join(
            c for c in unicodedata.normalize('NFD', string)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    def _train_line(self, category_tensor, line_tensor):
        hidden = self.rnn.init_hidden()

        self.rnn.zero_grad()

        for i in range(line_tensor.size()[0]): #for the length of the name
            output, hidden = self.rnn(line_tensor[i], hidden) #current character + hidden state

        loss = criterion(output, category_tensor)
        loss.backward()

        for p in self.rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    def _plot_training_losses(self, all_losses):
        plt.figure()
        plt.plot(all_losses)
        plt.show()

    def _read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self._unicode_to_ascii(line) for line in lines]

    def _letter_to_index(self, letter):
        return all_letters.find(letter)

    def _line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self._letter_to_index(letter)] = 1
        return tensor

    def _category_from_output(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def _random_choice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def _random_training_example(self):
        category = self._random_choice(self.all_categories)
        line = self._random_choice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self._line_to_tensor(line)
        return category, line, category_tensor, line_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)