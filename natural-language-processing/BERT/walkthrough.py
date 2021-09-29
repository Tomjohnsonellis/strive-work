import datasets


# HF has a very nice dataset tool: https://huggingface.co/docs/datasets/splits.html
# ag_dataset = datasets.load_dataset('ag_news', split=["train[:50%]","test[:50%]"])
ag_dataset = datasets.load_dataset("ag_news")
# It returns a data_dict, which we can examine like any python dictionary

print(ag_dataset)
print(ag_dataset["train"].features["label"])
print("-"*50) # We seem to have 4 news categories

# We load datasets in various ways, here's one of them:
# We could individually create train, test and validation datasets
ag_train_dataset = datasets.load_dataset('ag_news', split='train[:10%]') #10% of train
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
all_datasets = datasets.DatasetDict(ag_splits)
print(all_datasets)

# Now it's easy to access information we want like number of labels
num_labels = len(set(all_datasets['train'].unique('label') + 
                     all_datasets['test'].unique('label') +
                     all_datasets['val'].unique('label')))
print(f"Unique Labels: {num_labels}")
