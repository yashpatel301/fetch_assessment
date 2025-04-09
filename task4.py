import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel ##Importing tokenizer and model methods from Huggingface's transfomers module
import random


class MultiTaskModel(nn.Module):
    def __init__(self, backbone='bert-base-uncased', num_cls_labels=3, num_ner_labels=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(backbone) ##Retreive model we defined in above task
        hidden_size = self.bert.config.hidden_size ##Getting hidden layer's dim for input to our linear layer
        self.classifier = nn.Linear(hidden_size, num_cls_labels) ##Defining linear layer
        self.ner_head = nn.Linear(hidden_size, num_ner_labels) ##Defining linear layer

    def forward(self, input_ids, attention_mask, task):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if task == 'A':
            cls_token = outputs.last_hidden_state[:, 0, :] ##Getting cls token which will be considered as embeddings of a entire sentence
            return self.classifier(cls_token) ##Passing through linear layer
        else:
            token_embeds = outputs.last_hidden_state ##Getting cls token which will be considered as embeddings of a entire sentence
            return self.ner_head(token_embeds) ##Passing through linear layer

##First creating dataloader and dataset objects to fetch data in batches
class ClassificationDataset(Dataset):
    def __init__(self, tokenizer):
        self.sentences = [
            "He scored a goal.", "The minister spoke.", "Apple released a phone."
        ]
        self.labels = [0, 1, 2] ##sports, politics, tech
        self.encodings = tokenizer(self.sentences, truncation=True, padding=True, return_tensors='pt')  ##Tranformation of sentences to tensors

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx], ##Getting input ids of sentences
            'attention_mask': self.encodings['attention_mask'][idx], ##Getting attention mask of sentences
            'label': torch.tensor(self.labels[idx]) ##Getting labels of sentences
        }

class NERDataset(Dataset):
    def __init__(self, tokenizer):
        self.sentences = ["Barack Obama visited Berlin."]
        self.labels = [[1, 0, 0, 2, 0]] ##1 for B-PER, 0 for O, 2 for B-LOC

        self.encodings = tokenizer(self.sentences, padding='max_length', truncation=True, max_length=10, return_tensors='pt') ##Tranformation of sentences to tensors
        self.label_pad_token_id = -100 

        self.padded_labels = []
        for label in self.labels:
            padded = label + [self.label_pad_token_id] * (self.encodings['input_ids'].shape[1] - len(label)) ##Padding labels to match input ids length
            self.padded_labels.append(padded) ##Adding padded labels to list

    def __len__(self): return len(self.padded_labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx], ##Getting input ids of sentences
            'attention_mask': self.encodings['attention_mask'][idx], ##Getting attention mask of sentences
            'labels': torch.tensor(self.padded_labels[idx]) ##Getting labels of sentences
        }



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##I will use bert model where I can write insights at each step
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskModel().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) ##Defining optimizer

    cls_loader = DataLoader(ClassificationDataset(tokenizer), batch_size=1, shuffle=True) ##Creating dataloader for task A
    ner_loader = DataLoader(NERDataset(tokenizer), batch_size=1, shuffle=True)  ##Creating dataloader for task B
    cls_iter = iter(cls_loader) ##Creating iterator for task A
    ner_iter = iter(ner_loader) ##Creating iterator for task B

    
    model.train() ##Setting model to train mode
    ##Training loop for Task A
    for step in range(10):
        try:
            batch = next(cls_iter) ##Fetching batch from iterator
        except StopIteration:
            cls_iter = iter(cls_loader) ##Resetting iterator if end of dataset is reached
            batch = next(cls_iter) ##Fetching batch from iterator

        input_ids = batch['input_ids'].to(device) ##Getting input ids of sentences
        attention_mask = batch['attention_mask'].to(device) ##Getting attention mask of sentences
        labels = batch['label'].to(device) ##Getting labels of sentences

        logits = model(input_ids, attention_mask, task='A') ##Passing through model
        loss = F.cross_entropy(logits, labels) ##Calculating loss using cross entropy
        preds = torch.argmax(logits, dim=1) ##Getting predictions from logits
        acc = (preds == labels).float().mean().item() ##Calculating accuracy

        print(f"[Task A - Classification] Step {step} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

    ##Training loop for Task B
    for step in range(10):
        try:
            batch = next(ner_iter) ##Fetching batch from iterator
        except StopIteration:
            ner_iter = iter(ner_loader) ##Resetting iterator if end of dataset is reached
            batch = next(ner_iter) ##Fetching batch from iterator

        input_ids = batch['input_ids'].to(device) ##Getting input ids of sentences
        attention_mask = batch['attention_mask'].to(device) ##Getting attention mask of sentences
        labels = batch['labels'].to(device) ##Getting labels of sentences

        logits = model(input_ids, attention_mask, task='B')   ##Passing through model
        logits = logits.view(-1, logits.shape[-1]) ##Reshaping logits to match labels
        labels = labels.view(-1) ##Reshaping labels to match logits

        loss = F.cross_entropy(logits, labels, ignore_index=-100) ##Calculating loss using cross entropy
        preds = torch.argmax(logits, dim=-1) ##Getting predictions from logits
        mask = labels != -100 ##Creating mask to ignore padding tokens
        acc = (preds[mask] == labels[mask]).float().mean().item() ##Calculating accuracy

        print(f"[Task B - NER] Step {step} | Loss: {loss.item():.4f} | Token Acc: {acc:.2f}")
