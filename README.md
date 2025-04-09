# fetch_assessment

## Installation and triggering
To install all requirements: 

```
conda create -n fetch_assessment python==3.11
conda activate fetch_assessment
pip install -r requirements.txt
```

To run respective tasks: 
```
1. python task1.py
2. python task2.py
3. python task4.py
```

Task 3 has only textual context and can be access directly. 

## Descriptions and comments on scripts

**Task 1**
```
Insights to be drawn from inputs:
    -> We can see attention mask has been padded according to the longest statement from our example sentences.
    -> One thing to notice is first and last token index for all the statements are 101 and 102 which are the token indices for Start and End of statement.


Insights to be drawn from embeddings:
    -> First thing to notice is the shape of embeddings which says each token id has been converted to (768,) shape of float.
    -> Here we clearly have not given any indices needs to be emphasized and hense it says attentions are None.
```

**Task 2**
```
For this task, we will create a class to add layers at the end of our model for classification using PyTorch. 

Here based on sentence embeddings, classes will differ as per our defined classes. But since I am using pre-defined model, it will return classes based on its contextual understanding.

Changes:
    -> The idea remained same for Task2 as what I did in Task1. However, the only elements I added were the final layers linear and softmax to get logits and then probabilities for the sentences I defined.

    -> Also for classification, I took cls embeddings which are defined by the model for whole sentence and for NER, all token embeddings were passed separately to get each token classified.
```

**Task 4**
```
Key notes
    -> The model handles hypothetical data by manually constructing token-label pairs and ensuring label padding matches the tokenized sequence length, preventing shape mismatches during training. 
    -> The forward pass uses a shared BERT encoder with separate task-specific heads selected dynamically based on the current task. Metrics are calculated per task, using standard accuracy for classification and masked token-level accuracy for NER to exclude padding effects.
```