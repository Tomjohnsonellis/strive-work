"""
https://spacy.io/
spaCy is an industry standard natural language processing framework.
It's beginner friendly, but still powerful and highly optimised.
It's also free!

Corpus: A collection of writings or recorded remarks used for linguistic analysis

Pre-requisites: tokenisation (https://docs.google.com/presentation/d/1DjYDhQwKW9ztddA5R--cEr7mj2Ns-Kr5VY8_LJf6THQ/edit#slide=id.gcccab13826_0_7)
See here for an idea of how spaCy works: https://spacy.io/usage/linguistic-features

"""
#
import spacy
# Small
nlp = spacy.load("en_core_web_sm")
# Medium
# nlp = spacy.load("en_core_web_md")
# Large
# nlp = space.load("en_core_web_lg")

# No idea what these are yet!
# tagger and parser are the ones to learn about.
for thing in nlp.pipeline:
    print(thing)
print("-"*20)

# Let's do a "hello world"
text = "Hello, world. I am a simple sentence. My friend is eating apples but not bananas."
nlp_object = nlp(text)
# Printing the nlp_object will produce the original text
print(nlp_object)
# But it is much more than just a string! 
# It's a "spacy.tokens.doc.Doc", whatever that is
print(type(nlp_object))
# spacy has already done some tokenisation
print("-"*20, " Tokens")
for token in nlp_object:
    # These are actually "spacy.tokens.token.Token"
    # token.text would give you the raw string
    print(token)

# Splitting text isnt't very impressive, we could do that with just text.split,
# But spaCy can split SENTENCES!
print("-"*20, " Sentences")
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
print("-"*20, " Entity Recognition")
# Next, we'll have a look at the "ner" part of the nlp.pipeline
apple_text = "Apple builds tech products. I am eating an apple. Russia is the world's largest importer of apples."
apple_nlp = nlp(apple_text)
for entity in apple_nlp.ents:
    print(entity, entity.label_)
# It has detected that we meant "Apple" the company/ORGanisation, and fruit in other usages.
# It has also labelled Russia as a GPE or Geopolitical Entity.
# This is because spacy has been trained, nothing to do with grammar.

print("-"*20, " Lemmas")
# Next up, "lemmas", which are the infinitive form of a word (Running -> Run)
# Easiest way to explain lemmas is an example
lemma_nlp = nlp("I am walking and eating.")
for token in lemma_nlp:
    print(token.lemma_)
print([token.lemma_ for token in lemma_nlp])

print("-"*20, " Stop words")
# "Stop words" are words that have little value to a sentence.
# These are often ignored by search engines as they act as noise.
print(spacy.lang.en.stop_words.STOP_WORDS)
# This can be a good starting point for removing unecessary words,
# But best thing to do is define your own, problem-specific, stop words.

print("-"*20, " Stop words Cont.")
# Stop words can be removed from text like so:
dummy_text = "The fabulous statistics continued to pour out of the telescreen. As compared with last year there was more food, more clothes, more houses, more furniture, more cooking-pots, more fuel, more ships, more helicopters, more books, more babies—more of everything except disease, crime, and insanity. Year by year and minute by minute, everybody and everything was whizzing rapidly upwards. As Syme had done earlier Winston had taken up his spoon and was dabbling in the pale-coloured gravy that dribbled across the table, drawing a long streak of it out into a pattern. He meditated resentfully on the physical texture of life."
stop_nlp = nlp(dummy_text)
stops = [token.text for token in stop_nlp if token.is_stop]
useful_words = [token.text for token in stop_nlp if not token.is_stop]
# Quite a lot of words have been removed!
# print(stops)
# print(useful_words)
# For a quick count:
from collections import Counter
# print(Counter(stops))
# Can also find the top words!
print(Counter(stops).most_common(5))

print("-"*20, " Removing punctuation")
import string
print(f"Examples: {string.punctuation}")
# No spacy method here, we'll do it the normal pythonic way with a comprehension
text_without_punctuation = "".join([char for char in dummy_text if char not in string.punctuation])
removed_punctuation = "".join([char for char in dummy_text if char in string.punctuation])
#print(f"Text with no punctuation: {text_without_punctuation}")
print(f"Removed: {removed_punctuation}")

print("-"*20, " PartsOfSpeech (POS) Tagging")
# At some point in an English class, you may have learnt about Proper Nouns, Auxillary verbs, etc. This is those
speech_text = nlp("Alan Turing is regarded as one of the founding fathers of modern computing.")
print(speech_text)
# If you want to see what spacy thinks a word is, .pos_ will let you know, there are also tags!
# For a complete list of tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
print([(token, token.pos_, token.tag_) for token in speech_text])
# Thankfully, there is a more human readable version
for token in speech_text:
    print(token, " | ", token.pos_, " | ", token.tag_, " | ", spacy.explain(token.tag_))

print("-"*20, " Named Entity Recognition (NER)")
# For this one, please see ner_render.ipynb
# This is the more "real world" part, a spacy model is trained to detect what things in a sentence are.
ner_text = nlp("I am studying at Strive School.\nBig Ben is in London.\nJeff Bezos has a net worth of about $200 Billion.\nChickens lay eggs.\nPython is a reference to Monty Python, not the snake.")
# from spacy import displacy # Very funny.
# displacy.render(ner_text, style="ent")
# We can still have a look at the NER tags.
for entity in ner_text.ents:
    # The spacy.explain method has multiple uses
    print(entity.text, " | ", entity.label_, " | ", spacy.explain(entity.label_))







