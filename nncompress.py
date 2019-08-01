# -*- coding: utf-8 -*-
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as f
import torch.nn as nn
import torch
import random
import numpy as np
import time
import sys
import os


class PretrainedEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        """load pretrained embeddings, e.g., Glove """
        super(PretrainedEmbedding, self).__init__(
            *args, norm_type=2, **kwargs)
        self.vocab = {}
        self.i2w = {}

    def from_pretrained(self, file, freeze=True, ignore_first=True):
        i = 0
        with open(file, "r", encoding="utf-8") as f:
            print('Loading GloVe vectors...')
            pretrained_weight = np.zeros(
                (self.num_embeddings, self.embedding_dim))
            total = self.num_embeddings+1 if ignore_first else self.num_embeddings
            for line in f:
                if ignore_first and i == 0:
                    pass
                elif i < total:
                    cols = line.split()
                    word, vector = cols[0], [float(x) for x in cols[1:]]
                    assert len(vector) == self.embedding_dim
                    index = i-1 if ignore_first else i
                    self.vocab[word] = index
                    self.i2w[index] = word
                    pretrained_weight[index, :] = np.asarray(vector)
                    i += 1
                else:
                    break
        self.weight.data.copy_(torch.from_numpy(pretrained_weight))
        if freeze:

            self.weight.requires_grad = False

    # def get_code(self, embedding):
    #     with torch.no_grad():
    #         probs = self.forward(embedding)
    #         return [prob.argmax(dim=1) for prob in probs]


class EmbeddingCompressor(nn.Module):

    def __init__(self, embedding_dim, num_codebooks, num_vectors, use_gpu=False):
        r"""
        the main framework

        Args:
            embedding_dim: the dimission of word embeddings
            num_codebooks: number of codebooks (subcodes)
            num_vectors: number of vectors in each codebook
        """

        super(EmbeddingCompressor, self).__init__()
        self._tau = 1.0
        self.M = num_codebooks
        self.K = num_vectors

        self.use_gpu = use_gpu
        # E(w) -> h_w
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(embedding_dim,  self.M *
                      self.K // 2, bias=True),
            nn.Tanh()
        )
        # h_w -> a_w
        self.hidden_layer2 = nn.Linear(
            self.M * self.K // 2,  self.M * self.K, bias=True)
        self.codebook = nn.Parameter(torch.FloatTensor(
            self.M * self.K, embedding_dim), requires_grad=True)

    def _encode(self, embeddings):
        # E(w) -> h_w   ~[B, M*K//2]
        h = self.hidden_layer1(embeddings)
        # h_w -> a_w    ~[B, M * K]
        logits = f.softplus(self.hidden_layer2(h))

        # ~[B, M, K]
        logits = logits.view(-1, self.M, self.K).contiguous()
        return logits

    def _decode(self, gumbel_output):
        return gumbel_output.matmul(self.codebook)

    def forward(self, vector):
        # 1. Encoding
        logits = self._encode(vector)
        # ~[B, M, K]
        # 2. Discretization
        # a_w -> d_w    ~[B, M, K]
        D = f.gumbel_softmax(
            logits.view(-1, self.K).contiguous(), tau=self._tau, hard=False)

        gumbel_output = D.view(-1, self.M*self.K).contiguous()
        maxp, _ = D.view(-1, self.M, self.K).max(dim=2)
        # 3. Decoding
        gumbel_output = f.layer_norm(gumbel_output, gumbel_output.size())
        pred = self._decode(gumbel_output)  # y_hat
        return logits, maxp.data.clone().mean(), pred


class Trainer:
    def __init__(self, model, num_embeddings, embedding_dim, model_path, lr=1e-4, use_gpu=False, batch_size=64):

        self.model = model
        self.embedding = PretrainedEmbedding(num_embeddings, embedding_dim)
        self.vocab_size = len(self.embedding.vocab)
        self.use_gpu = use_gpu
        self._batch_size = batch_size
        self.optimizer = Adam(model.parameters(), lr=lr)
        self._model_path = model_path

    def load_pretrained_embeddings(self, file, freeze=True, ignore_first=False):
        self.embedding.from_pretrained(file, freeze, ignore_first)
        self.vocab_size = len(self.embedding.vocab)

    def run(self, max_epochs=200):
        """Train the model by compressing embeddings and save the model to `self._model_path`. 
        """
        torch.manual_seed(3)
        criterion = nn.MSELoss(reduction="sum")
        valid_ids = torch.from_numpy(np.random.randint(
            0, self.vocab_size, (self._batch_size * 10,))).long()
        # Training
        # Initialize variables
        best_loss = float('inf')
        vocab_list = [x for x in range(self.vocab_size)]
        for epoch in range(max_epochs):
            self.model.train()
            start_time = time.time()
            random.shuffle(vocab_list)
            train_loss_list = []
            train_maxp_list = []
            for start_idx in range(0, self.vocab_size, self._batch_size):

                word_ids = torch.Tensor(
                    vocab_list[start_idx:start_idx + self._batch_size]).long()
                self.optimizer.zero_grad()
                input_embeds = self.embedding(word_ids)
                if self.use_gpu:
                    input_embeds = input_embeds.cuda()
                logits, maxp, pred = self.model(input_embeds)
                loss = criterion(pred, input_embeds).div(self._batch_size)
                train_loss = loss.data.clone().item()
                train_loss_list.append(train_loss)
                train_maxp_list.append(
                    maxp.cpu() if self.use_gpu else maxp)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.001)
                self.optimizer.step()
            # Print every epoch
            time_elapsed = time.time() - start_time
            train_loss = np.mean(train_loss_list)/2
            train_maxp = np.mean(train_maxp_list)
            # Validation
            self.model.eval()
            valid_loss_list = []
            valid_maxp_list = []
            for start_idx in range(0, len(valid_ids), self._batch_size):
                word_ids = valid_ids[start_idx:start_idx +
                                     self._batch_size]
                oracle = self.embedding(word_ids)
                if self.use_gpu:
                    oracle = oracle.cuda()
                logits, maxp, pred = self.model(oracle)
                loss = criterion(pred, oracle).div(self._batch_size)
                valid_loss = loss.data.clone().item()
                valid_loss_list.append(valid_loss)
                valid_maxp_list.append(
                    maxp.cpu() if self.use_gpu else maxp)
            # Report
            valid_loss = np.mean(valid_loss_list)/2
            valid_maxp = np.mean(valid_maxp_list)
            if train_loss < best_loss * 0.99:
                best_loss = train_loss
                print("[epoch{}] trian_loss={:.2f}, train_maxp={:.2f}, valid_loss={:.2f}, valid_maxp={:.2f},  bps={:.0f} ".format(
                    epoch, train_loss, train_maxp,
                    valid_loss, valid_maxp,
                    len(train_loss_list) / time_elapsed
                ))
        print("Training Done")

    def export(self, prefix, sample_words=[]):
        """Export word codes and codebook for given embeddings.
        Args:
            prefix: the path prefix to save files
        """
        assert os.path.exists(self._model_path + ".pt")
        vocab_list = list(range(self.vocab_size))
        # Dump codebook
        codebook = dict(self.model.named_parameters())["codebook"].data
        if self.use_gpu:
            codebook = codebook.cpu()
        np.save(prefix + ".codebook",
                codebook.numpy())
        # Dump codes
        text = ""
        sample_words = set(sample_words)
        with open(prefix + ".codes", "w", encoding="utf8") as fout:
            vocab_list = list(range(self.vocab_size))
            for start_idx in tqdm(range(0, self.vocab_size, self._batch_size)):
                word_ids = torch.Tensor(
                    vocab_list[start_idx:start_idx + self._batch_size]).long()
                # Coding
                input_embeds = self.embedding(word_ids)
                if self.use_gpu:
                    input_embeds = input_embeds.cuda()
                logits = self.model._encode(input_embeds)
                _, codes = logits.max(dim=2)
                for wid, code in zip(word_ids, codes):
                    # cuda to int/list
                    wid = wid.item()
                    if self.use_gpu:
                        code = code.data.cpu().tolist()
                    else:
                        code = code.data.tolist()
                    word = self.embedding.i2w[wid]
                    if word in sample_words:
                        text += word + "\t" + " ".join(map(str, code)) + "\n"
                    fout.write(word + "\t" +
                               " ".join(map(str, code)) + "\n")
        if text:
            print(text)

    def evaluate(self):
        assert os.path.exists(self._model_path + ".pt")
        vocab_list = list(range(self.vocab_size))
        distances = []
        for start_idx in range(0, self.vocab_size, self._batch_size):
            word_ids = torch.Tensor(
                vocab_list[start_idx:start_idx + self._batch_size]).long()
            input_embeds = self.embedding(word_ids)
            if self.use_gpu:
                input_embeds = input_embeds.cuda()
            _, _, reconstructed = self.model(input_embeds)
            distances.extend(np.linalg.norm(
                (reconstructed-input_embeds).data.cpu(), axis=1).tolist())
        return np.mean(distances)
