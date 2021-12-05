import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import numpy as np

import os
import time
import copy
import shutil
import pathlib
import argparse
import json

import numpy as np


class Classifier(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim):
        super(Classifier, self).__init__()


        # Embedding is just an lookup table of size "vocab_size"
        # and each element has "embedding_size" dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Shape --> [Sequence_length , batch_size , embedding dims]
        embedding = self.embedding(x)
        # Shape --> (output) [Sequence_length , batch_size , hidden_size]
        # Shape --> (hs, cs) [num_layers, batch_size size, hidden_size]
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)
        
        linear_outputs = self.fc(hidden_state)
        
        return linear_outputs

    
class LangDataset(Dataset):
    def __init__(self, ds, train_vocab=None):
        self.corpus = ds

        if not train_vocab:
            self.src_vocab, self.trg_vocab = self._build_vocab()
        else:
            self.src_vocab, self.trg_vocab = train_vocab

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, item):
        text = self.corpus.iloc[item].Text
        lang = self.corpus.iloc[item].Language
        
        return {
            'src': self.src_vocab.lookup_indices(text.lower().split()),
            'trg': self.trg_vocab.lookup_indices([lang])
        }
    
    def _build_vocab(self):
        src_tokens = self.corpus.Text.str.cat().lower().split()

        src_vocab = build_vocab_from_iterator([src_tokens], specials=["<unk>","<pad>"])
        src_vocab.set_default_index(src_vocab['<unk>'])

        trg_vocab = build_vocab_from_iterator([['English', 'French', 'Spanish']])
        
        return src_vocab, trg_vocab

    
def collate_fn(batch, pad_value, device):
    trgs = []
    srcs = []
    for row in batch:
        srcs.append(torch.tensor(row["src"], dtype=torch.long).to(device))
        trgs.append(torch.tensor(row["trg"]).to(device))

    padded_srcs = pad_sequence(srcs, padding_value=pad_value)
    padded_trgs = pad_sequence(trgs, padding_value=pad_value)
    return {"src": padded_srcs, "trg": padded_trgs}

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained model
    pre_model = torch.load(os.path.join(model_dir, 'classifier-model.pth'), map_location=device)
    model = Classifier(pre_model["len_src_vocab"], pre_model["len_src_vocab"], 125, 2)
    model.load_state_dict(pre_model["model_state_dict"])
    
    return {
        "model": model.to(device),
        "src_vocab": pre_model["src_vocab"]
    }

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        if not isinstance(data, str):
            raise ValueError("Unsupported input type. Input type can only be a string. \
                             I got {}".format(data))
                       
        return data
    raise ValueError("Unsupported content type: {}".format(request_content_type))
    
def predict_fn(input_fn_out, model_fn_out):
    out_map = {0: 'English', 1: 'French', 2: 'Spanish'}
    model = model_fn_out["model"]
    model.eval()
    
    txt_to_ind = model_fn_out["src_vocab"].lookup_indices(input_fn_out.split())
    ind_tensors = torch.tensor(txt_to_ind).unsqueeze(1)
    
    with torch.no_grad():
        res = model(ind_tensors).view(-1).argmax().item()
    
    return out_map[res]
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    
    # These variables are populate with estimator hypermeters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=125)
    parser.add_argument("--hidden-dim", type=int, default=2)
    
    args, _ = parser.parse_known_args()
    
    # Estimator's config "output_path" populates `SM_OUTPUT_DATA_DIR` enviroment variable
    model_storage = os.environ['SM_OUTPUT_DATA_DIR']
    
    # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#prepare-a-pytorch-training-script
    # estimator.fit populates `SM_CHANNEL_TRAIN` enviroment variable
    corpus_dir = os.environ['SM_CHANNEL_TRAIN']
    
    ds = pd.read_csv(corpus_dir + "/Language Detection.csv")
    
    np.random.shuffle(ds.values)
    total_ds = ds.loc[(ds['Language'] == 'Spanish') | (ds['Language'] == 'French') | (ds['Language'] == 'English')]
    
    train_ds = total_ds[:2700]
    valid_ds = total_ds[2700:]
    
    train_langds = LangDataset(train_ds)
    valid_langds = LangDataset(valid_ds, (train_langds.src_vocab, train_langds.trg_vocab))

    pad_value = train_langds.src_vocab['<pad>']

    train_dt = DataLoader(train_langds, batch_size=args.batch_size, shuffle=
                       True, collate_fn=lambda batch_size: collate_fn(batch_size, pad_value, device))

    valid_dt = DataLoader(valid_langds, batch_size=args.batch_size, shuffle=
                       True, collate_fn=lambda batch_size: collate_fn(batch_size, pad_value, device))
    
    model = Classifier(len(train_langds.src_vocab), 
                       len(train_langds.src_vocab), 
                       args.embedding_dim, 
                       args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        print('Epoch: ', epoch)
        for idx, batch in enumerate(train_dt):
            src = batch["src"]  # shape --> e.g. (19, 2) sentence len, batch size
            trg = batch["trg"]  # shape --> e.g. (3, 2) sentence len, batch size

            # Clear the accumulating gradients
            optimizer.zero_grad()

            # shape --> (1, 32, 3) 1, batch size, trg vocab
            output = model(src)

            # Calculate the loss value for every epoch
            # Squeezing to remove first dimension 
            loss = criterion(output.squeeze(0), trg.squeeze(0))

            # Calculate the gradients for weights & biases using back-propagation
            loss.backward()

            epoch_loss += loss.item()

            # Clip the gradient value is it exceeds > 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Update the weights values
            optimizer.step()
        print('\tTrain loss: ', epoch_loss/len(train_dt))

        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dt):
                src = batch["src"]  # shape --> e.g. (19, 2) sentence len, batch size
                trg = batch["trg"]  # shape --> e.g. (3, 2) sentence len, batch size

                output = model(src)

                # Calculate the loss value for every epoch
                loss = criterion(output.squeeze(0), trg.squeeze(0))

                epoch_loss += loss.item()

        print('\tEval loss: ', epoch_loss/len(valid_dt))
    
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'len_src_vocab': len(train_langds.src_vocab),
        'src_vocab': train_langds.src_vocab,
        'len_trg_vocab': len(train_langds.trg_vocab),
        'trg_vocab': train_langds.trg_vocab
    }, model_storage + "/classifier-model.pth")
    
    

