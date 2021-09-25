# Embeddings are a fundamental concept for NLP

# N-grams
# A sequence of N items from a given linguistic sequence
# E.g. Bi-grams of "Apple" could be "Ap, pp, pl, le"


# Vector space models
# An item (word, sentence, other) can be represented as a vector of numbers
# Not as in ABC is [123] but more as a vector of it's occurences in a corpus.
# Trigrams (https://en.wikipedia.org/wiki/Trigram) are often used.
# A trigram of banana could be something like "#ba", "nan" or "na#", # here repesents the end of the word.
# Say we had a vector for the word "banana", it could contain the frequencies of each 3 letter combination

# Relatedness
# One of reasons we'd use vectors (other than the fact that NNs cannot make use of just text)
# is that we can then compare vectors to look for similarities.
# This is done using the ~ Cosine Similarity ~, which is the cosine of the angle between two vectors in some n-dimensional space.
# It's a bit abstract, check out: https://deepai.org/machine-learning-glossary-and-terms/cosine-similarity
# Why use it? Counting occurences is another method to see how two things are similar, 
# but doesn't work too well for comparing small corpora to large ones.
# Example: We humans can see that a & b are a bit similar, how can we express this mathematically?
import spacy
nlp_model = spacy.load("en_core_web_sm")
document_a = nlp_model("Italy is the home of pizza")
document_b = nlp_model("Rome is the capital of Italy")
document_c = nlp_model("My spoon is too big")
documents = [document_a, document_b, document_c]
for doc in documents:
    print(document_a.similarity(doc))
# Generally: ~1 means very similar, ~0 is unrelated, 
# ~ -1 is similar but opposite e.g. Paris, France / Rome, Italy - Both cities but different countries.
# There IS a relation, but it's not just them being similar.

# The use of trigrams means that the vocabulary for a document quickly becomes massive
# and the vector space becomes extremely high-dimensional and difficult to compute.
# This is where embeddings come in!
# Embeddings are a way to reduce dimensions and make the problem space more manageable
# Think of it like: "man is to woman as king is to queen",
# Man and King are very different words in isolation, but have similar contextual meaning to the female versions
# Word2Vec is a tool developed by Google for this, it is based on the idea that
# Words with similar surrounding words are often semantically similar.
# E.g. "I took the bus to work" / "I took the train to work", bus/train are both forms of transport
# This is a decent guide to the concepts https://jalammar.github.io/illustrated-word2vec/