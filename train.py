import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import X_tokenize,y,tags
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# import torch
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW,BertForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("./PhoBERT_base_transformers/dict.txt")



MAX_LEN = 125

train_ids = []
for sent in X_tokenize:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    train_ids.append(encoded_sent)

train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

train_masks = []
for sent in train_ids:
    mask = [int(token_id > 0) for token_id in sent]
    train_masks.append(mask)
    
y_train =[]
for i in y:
  label = tags.index(i)
  y_train.append(label)
# y_train

train_inputs = torch.tensor(train_ids)
train_masks = torch.tensor(train_masks)
y_trains = torch.tensor(y_train)

train_data = TensorDataset(train_inputs, train_masks, y_trains)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)


config = RobertaConfig.from_pretrained(
    "./PhoBERT_base_transformers/config.json", from_tf=False, num_labels = 14, output_hidden_states=False,
)
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "./PhoBERT_base_transformers/model.bin",
    config=config
)
BERT_SA.cuda()


#training
import random
from tqdm import tqdm_notebook
device = 'cpu'
epochs = 100

param_optimizer = list(BERT_SA.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    BERT_SA.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0
    
    for step, batch in tqdm_notebook(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        BERT_SA.zero_grad()
        outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        nb_train_steps += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
        optimizer.step()   
    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))
print("Training complete!")



data = {
"model_state": BERT_SA.state_dict(),
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')