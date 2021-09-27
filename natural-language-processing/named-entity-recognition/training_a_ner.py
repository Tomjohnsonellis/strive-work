"""
Named Entity Recognition: NER
A process where a sentence or chunk of text is parsed through to find entities that can be put into categories.
(Person, Place, Organisation...)

spaCy has some pretrained models that we can refine to our needs.
Today we will be designing a new label and training a model to recognise it.

"""

import spacy
import random
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

# Load an NLP model
nlp = spacy.load("en_core_web_md")

# We will try and train the model to recognise "foods"
# So load up the food data
with open("natural-language-processing/named-entity-recognition/data/food.txt") as foodfile:
    foods = foodfile.read()

doc = nlp(foods)

# Take a look at the current classifications if you like, they aren't the best!
print("Entities: ",[(ent.text, ent.label_) for ent in doc.ents])
print("-"*50)

# We will need to have some training data, in this case just a bunch of different food items
words = ["ketchup", "pasta", "carrot", "pizza",
         "garlic", "tomato sauce", "basil", "carbonara",
         "eggs", "cheek fat", "pancakes", "parmigiana", "eggplant",
         "fettucine", "heavy cream", "polenta", "risotto", "espresso",
         "arrosticini", "spaghetti", "fiorentina steak", "pecorino",
         "maccherone", "nutella", "amaro", "pistachio", "coca-cola",
         "wine", "pastiera", "watermelon", "cappuccino", "ice cream",
         "soup", "lemon", "chocolate", "pineapple"]

# We will use this to create a training dataset
# Go through the food file, clean up the text a bit

# Preprocessing - Make everything lowercase
words = [word.lower() for word in words]
with open("natural-language-processing/named-entity-recognition/data/food.txt") as foodfile:
    food_dataset = foodfile.readlines()
food_dataset = [sentence.lower() for sentence in food_dataset]


# Extract elements for the training data
# SpaCy requires data in the form of: ("This is a sentence about food", {"entities":[25,29,"FOOD"]})
train_data = []

for sentence in food_dataset:
    entities = []
    for word in words:
        if word in sentence:
            # Find their indexes, the character positions of each word. 
            # "red" in "the red cat" would be (4,7)
            start_index = sentence.index(word)
            end_index = len(word) + start_index
            # Also add the label while we're here
            pos = (start_index, end_index, "FOOD")
            # Add it to the entities list
            entities.append(pos)
    # Store the sentence and accompanying food locations
    element = (sentence.rstrip("\n"), {"entities":entities})
    # Add that to our training data
    train_data.append(element)


# Note: There are better ways to get good training data, this is just a "from-scratch" method


# Next up, update the spaCy model with our new data
named_entity_recogniser = nlp.get_pipe("ner")
# Add our new "food" label
named_entity_recogniser.add_label("FOOD")

# In order to speed up model training, we'll use spacy's disable_pipes method, so we only alter what we need to.
things_to_affect = ["ner", "trf_wordpiecer","trf_tok2vec"]
unaffected = [pipe for pipe in nlp.pipe_names if pipe not in things_to_affect]
with nlp.disable_pipes(*unaffected):
    for iteration in range(10):
        # Shuffle the data, to help prevent overfitting
        random.shuffle(train_data)
        losses = {}

        # Batch up the data
        batches = minibatch(train_data, size=8)
        # Work through each batch
        for index, batch in enumerate(batches):
            for text, entities in batch:
                # Make a doc out of the text
                doc = nlp.make_doc(text)
                # Create a spaCy "Example", which is two Docs, one is the correctly labelled info, the other is the pipeline's predictions
                example = Example.from_dict(doc, entities)
                # Update our nlp model with the info
                nlp.update([example], losses=losses, drop=0)
            print(f"Iteration: {iteration} | Batch: {index} | Loss: {losses}")

# After training a model, we'd like to save it.
nlp.to_disk("natural-language-processing/named-entity-recognition/data/trained.nlp")
print("Saved!")

# For the sake of not retraining the model on each test...
if __name__ == "__main__":
    # Let's test it:
    trained_nlp = spacy.load("natural-language-processing/named-entity-recognition/data/trained.nlp")
    # A sentence from the training set
    test_one = trained_nlp("Get your hands off my pancakes!")
    # Two new sentences containing known foods
    test_two = trained_nlp("The italian chef recently pasta way.")
    test_three = trained_nlp("Is wine even real? Grape soup is not.")
    # Terms not in the training data
    test_four = trained_nlp("chicken hamburger caviar")

    tests = [test_one, test_two, test_three, test_four]
    for doc in tests:
        print("Entities: ", [(entity.text, entity.label_) for entity in doc.ents])



            
