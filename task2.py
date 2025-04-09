import torch ##Importing pytorch
from torch import nn
from transformers import AutoTokenizer, AutoModel ##Importing tokenizer and model methods from Huggingface's transfomers module


class task2Model(nn.Module):
    def __init__(self, model, output_dim1, output_dim2):
        super(task2Model, self).__init__()
        self.model = model ##Retreive same model we defined in above task
        hidden_size = self.model.config.hidden_size ##Getting hidden layer's dim for input to our linear layer
        self.linear = nn.Linear(hidden_size, output_dim1) ##Defining linear layer
        self.softmax = nn.Softmax(dim=1) ##Defining softmax layer to fetch probabilities from logits
        self.linear2 = nn.Linear(hidden_size, output_dim2) ##Defining linear layer
        self.softmax2 = nn.Softmax(dim=1) ##Defining softmax layer to fetch probabilities from logits

    def forward(self, sentences, task='A'):
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True) ##Passing our sentences to input
        
        with torch.no_grad(): 
            outputs = model(**inputs)
        
        if task=='A':
            embeds = outputs.last_hidden_state[:, 0, :] ##Getting cls token which will be considered as embeddings of a entire sentence
            lin_out = self.linear(embeds) ##Passing through linear layer
            probs = self.softmax(lin_out) ##returning after passing through softmax layer
            predictions = torch.argmax(probs, dim=1)
            return predictions
        else:
            token_embeds = outputs.last_hidden_state ##Getting cls token which will be considered as embeddings of a entire sentence
            lin_out = self.linear2(token_embeds) ##Passing through linear layer
            return self.softmax2(lin_out) ##returning after passing through softmax layer

if __name__ == "__main__":
    ##I will use bert model where I can write insights at each step
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    sentences = [
        "The team won the championship after a tough season.",
        "The government passed a new bill today.",
        "A new AI model outperforms previous systems."
    ]
    label_map = {0: 'sports', 1: 'politics', 2: 'technology'}
    labels = torch.tensor([0, 1, 2])
    
    sentence_classification_model = task2Model(model, 2, 3) ##Defining our model with output dimension of 2 and 3 for task A and B respectively
    predictions = sentence_classification_model.forward(sentences, task='A')
    for sentence, pred in zip(sentences, predictions):
        print(f"Sentence: '{sentence}' → Predicted Class: {label_map[pred.item()]}")

    sentences = [
        "Barack Obama was the 44th President of the United States.",
        "Google is headquartered in Mountain View, California.",
        "The Eiffel Tower is located in Paris, France."
    ]
    print(sentence_classification_model(sentences, task='B')) ##Getting predictions for task B
