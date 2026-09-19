"""Bidirectional LSTM classifier and its PyTorch dataset wrapper."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """Thin wrapper so PyTorch can iterate over numpy arrays of sequences and labels in batches."""

    def __init__(self, sequences, labels):
        # LongTensor because embedding layers expect integer (long) indices
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier with a small feed-forward head on top."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        # each word id -> dense vector; padding_idx=0 keeps <PAD> pinned at zero
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # stacked bidirectional LSTM; batch_first gives input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # hidden_dim*2 because a bidirectional LSTM outputs a forward and a backward state
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                         # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)
        # concatenate the final forward and backward hidden states for the classifier head
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)                                 # logits for the output classes
