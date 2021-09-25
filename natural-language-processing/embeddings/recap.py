"""
Terms:
Corpus - A body of text. Wikipedia article, tweets, diary entries...
Token - An element (word, sentence) present in a corpus
Tokenization - The process of converting text into tokens
Stop word - A word that adds little information to the text, "this", "do", "more", "such"
Stemming - A simple process of removing typical pre/suffixes from a word to try and find the "base" of it
Lemmatization - In most cases, better stemming. Uses a combination of grammar rules, word structure and detailed dictionary entries.

Text Preprocessing
There are a few steps that are almost always going to be helpful when working with text.
Noise Removal - Get rid of any unneeded things like punctuation, URLs, stopwords
Lexicon Normalization
Object Standardization

"""
# Noise Removal
def remove_noise(list_of_words):

    symbols = "!\"£$%^&*()\',./#;:[]"

    # Go through each word
    for index, word in enumerate(list_of_words):
        # And remove any of the above symbols that appear
        for symbol in symbols:
            list_of_words[index] = list_of_words[index].replace(symbol, "")


    return list_of_words

some_words = ["Hello!","My","name","is","Jenkins,","I'm","(almost)","worth","£1,000,000!"]
print(remove_noise(some_words))
