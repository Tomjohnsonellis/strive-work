"""
https://spacy.io/
spaCy is an industry standard natural language processing framework.
It's beginner friendly, but still powerful and highly optimised.
It's also free!

Pre-requisites: tokenisation (https://docs.google.com/presentation/d/1DjYDhQwKW9ztddA5R--cEr7mj2Ns-Kr5VY8_LJf6THQ/edit#slide=id.gcccab13826_0_7)
See here for an idea of how spaCy works: https://spacy.io/usage/linguistic-features

"""
#
import spacy
nlp = spacy.load("en_core_web_sm")
# Lightweight alternative
# nlp = space.load("en_core_web_lg")

# No idea what these are yet!
# tagger and parser are the ones to learn about.
for thing in nlp.pipeline:
    print(thing)
print("-"*50)

# Let's do a "hello world"
text = "Hello, world. I am a simple sentence. My friend is eating apples but not bananas."
nlp_object = nlp(text)
# Printing the nlp_object will produce the original text
print(nlp_object)
# But it is much more than just a string! 
# It's a "spacy.tokens.doc.Doc", whatever that is
print(type(nlp_object))
# spacy has already done some tokenisation
print("-"*50)
for token in nlp_object:
    # These are actually "spacy.tokens.token.Token"
    # token.text would give you the raw string
    print(token)

# Splitting text isnt't very impressive, we could do that with just text.split,
# But spaCy can split SENTENCES!
print("-"*50)
print(nlp_object.sents) # "sents" is another generator object
for sentence in nlp_object.sents:
    # Again, this is a spacy object, "spacy.tokens.span.Span" in this case
    # sentence.text would be the raw string
    print(sentence)
# Can also get the sentences like this...
print( list(nlp_object.sents) ) # And access each "sentence object" the list way

# Note: If you are only interested in the tokeniser part of spacy,
# This will save you some memory
tokeniser = nlp.tokenizer(text)
# for token in tokeniser:
#     print(token)


# You will need the "ner" part of the pipeline for this, attempting to only use
# a tokenizer will not throw an error, but there will be no information.
print("-"*50)
# Next, we'll have a look at the "ner" part of the nlp.pipeline
apple_text = "Apple builds tech products. I am eating an apple. Russia is the world's largest importer of apples."
apple_nlp = nlp(apple_text)
for entity in apple_nlp.ents:
    print(entity, entity.label_)
# It has detected that we meant "Apple" the company/ORGanisation, and fruit in other usages.
# It has also labelled Russia as a GPE or Geopolitical Entity.
# This is because spacy has been trained, nothing to do with grammar.

