import pandas as pd
import numpy as np
import torchtext
import spacy

def build_dataset(iterator, max_size=None):
    if not max_size:
        dataset = []
        for label, line in iterator:
            tokens = []
            tokens += tokenize(label, line)
            dataset.append([label, tokens])
        df = pd.DataFrame(dataset, columns=["rating","tokens"])
    if max_size:
        count = 0
        dataset = []
        for label, line in iterator:
            tokens = []
            tokens += tokenize(label, line)
            dataset.append([label, tokens])
            count += 1
            if count == max_size:
                break
        df = pd.DataFrame(dataset, columns=["rating","tokens"])

    return df

def tokenize(label, line):
    return line.split()

def adjust_ratings(amazon_df):
    amazon_df["rating"] = amazon_df["rating"].subtract(1)

    return amazon_df


def count_words(tokens):
    vocab = {}
    for sentence in tokens:
        for token in sentence:
            if token not in vocab:
                vocab.update({token:1})
            else:
                vocab[token] += 1

    vocab_list = list(vocab.items())
    sorted_vocab = sorted((value, key) for (key, value) in vocab_list)

    return sorted_vocab


def create_stop_words(vocab, count=20):
    stop_words = [word for _, word in vocab[-count:]]
    return stop_words

def remove_punctuation(df):
    symbols = "!\"£$%^&*()\',./#;:[]"
    for symbol in symbols:
        df["tokens"] = df["tokens"].apply(lambda token_set: [token.replace(symbol, "") for token in token_set])
    
    return df



if __name__ == "__main__":
    amazon = torchtext.datasets.AmazonReviewFull
    training_iterator = amazon(split="test") # The training dataset is massive, the test will be MORE than enough for us
    # Load the dataset, 650,000 entries by default, I'm going to use less
    df = build_dataset(training_iterator, 10000)
    # The ratings are 1 to 5 by default, we will convert to 0 to 4 to be more pythonic
    df = adjust_ratings(df)

    # TODO: Preprocessing? Remove stop words and punctuation, convert to consistent case
    # Lemmatise?

    # Make lowercase
    df["tokens"] = df["tokens"].apply(lambda token_set: [token.lower() for token in token_set])

    # Remove punctuation
    df = remove_punctuation(df)

    # Stop word removal
    vocab = count_words(df["tokens"])
    print(f"Unique words: {len(vocab)}")
    # I'll make the 20 most used words my stop-words, looking into the data that seems plausibly useful
    stop_words = create_stop_words(vocab, 20)
    # Remove them from the dataset
    # Lambdas were a bit confusing for me, but I think I get them now
    # This removes any tokens that appear in our stop-words
    df["tokens"] = df["tokens"].apply(lambda token_set: [token for token in token_set if token not in stop_words])





    # Now that we have a nice dataset, we can begin the natural language processing
    # Load a model that you can run on the current hardware
    # nlp = spacy.load("en_core_web_sm") # Small model
    # nlp = spacy.load("en_core_web_md") # Medium model
    # nlp = space.load("en_core_web_lg") # Large model




    