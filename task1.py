import torch ##Importing pytorch
from transformers import AutoTokenizer, AutoModel ##Importing tokenizer and model methods from Huggingface's transfomers module

##I will use bert model where I can write insights at each step
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

##Defining some example statements to get embeddings using the encoder of our model. 
example_sent = ["The sky is blue", "Tom like chocolates but he can't eat it because of diabetes", "Spring is about to end and summer will come soon."]

##Tokenizing and passing tokenized list to model
inputs = tokenizer(example_sent, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
  outputs = model(**inputs)

print(f'Input: {inputs}')

attention = inputs['attention_mask']

print(outputs.last_hidden_state.shape)
print(outputs)
