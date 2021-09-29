import datasets
import numpy as np
# This is an example of using BERT.
# For an explanation of BERT I'd suggest https://www.youtube.com/watch?v=xI0HHN5XKDo
# Or this paper if you prefer those: https://arxiv.org/pdf/1810.04805.pdf

# HF has a very nice dataset tool: https://huggingface.co/docs/datasets/splits.html
# ag_dataset = datasets.load_dataset('ag_news', split=["train[:50%]","test[:50%]"])
ag_dataset = datasets.load_dataset("ag_news")
# It returns a data_dict, which we can examine like any python dictionary

print(ag_dataset)
print(ag_dataset["train"].features["label"])
print("-"*50) # We seem to have 4 news categories

# We load datasets in various ways, here's one of them:
# We could individually create train, test and validation datasets
ag_train_dataset = datasets.load_dataset('ag_news', split='train[:5%]') #5% of train
ag_test_dataset = datasets.load_dataset('ag_news', split='test[11%:12%]') # A specific 1% of test
ag_val_dataset = datasets.load_dataset('ag_news', split='train[10%:11%]') # A 1% of train that we aren't already using

# And then merge them into one with the DatasetDict() method
# First, combine what we have:
ag_splits = {split: data for split, data in zip(['train', 'test', 'val'], [ag_train_dataset, ag_test_dataset, ag_val_dataset])}
"""
This is making a dictionary like:
    {
    train:<the_training_data>,
    test:<the_test_data>,
    val:<the_validation_data>
    }
"""
print("Zipped:")
print(ag_splits)
print("-"*50) # We seem to have 4 news categories

# Then turn into a DatasetDict(), which looks quite similar
all_data = datasets.DatasetDict(ag_splits)
print(all_data)

# Now it's easy to access information we want like number of labels
# (We will need this later)
num_labels = len(set(all_data['train'].unique('label') + 
                     all_data['test'].unique('label') +
                     all_data['val'].unique('label')))
print(f"Unique Labels: {num_labels}")
# See functions.py for this process as a function

########
# Next up, some preprocessing
# As we are working with text, we will need to tokenize the data
from transformers import AutoTokenizer
pretrained_bert = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
# We can write our own small function to help out
def tokenize(dataset):
    # Grab the sentences from a split
    sentences = dataset["text"]
    # Apply the tokenizer to them
    return tokenizer(sentences, padding="max_length", truncation=True)

# Then apply it with a mapping
tokenized_dataset = all_data.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenize data")
print("-"*50)
print(tokenized_dataset)

# Great! We now have a tokenized dataset, now we can create a model
# We will use huggingface's already available things here
from transformers import AutoConfig
config = AutoConfig.from_pretrained(pretrained_bert, num_labels=num_labels)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert, config=config)

# When using pre-trained models, we do not train as such, but fine-tune
# We actually only train the classification part, BERT is a fine model as is
from transformers import TrainingArguments, Trainer
import torch
# Look this up if you're interested
training_args = TrainingArguments(output_dir='bert-ag-news-classification',
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=2,
                                  num_train_epochs=3,
                                  max_steps=10,
                                  logging_steps=100,
                                  logging_dir='bert-ag-news-classification/tb',
                                  evaluation_strategy='epoch',
                                  no_cuda=not torch.cuda.is_available()
                                  )
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=training_args,
                  train_dataset=tokenized_dataset['train'],
                  eval_dataset=tokenized_dataset['val'])

print("@"*50)

# I do not have the hardware to run this unfortunately, 
# but the training command is nice and simple!
train_result = trainer.train()
# The trainer also has some other helpful methods
trainer.save_model()  
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
# After saving the model, we can evaluate it!
eval_metrics = trainer.evaluate()
trainer.log_metrics("val", eval_metrics)
trainer.save_metrics("val", eval_metrics)
trainer.save_state()
# And show some results in a human-readable format
# Make some predictions
test_result = trainer.predict(test_dataset=tokenized_dataset['test'])
# Convert the logits to label numbers
predicted_label_ids = np.argmax(test_result.predictions, axis=1)
# Find out what those label numbers mean
predicted_label_names = tokenized_dataset['test'].features['label'].int2str(predicted_label_ids)
# Display!
print(predicted_label_names)




