"""
Named Entity Recognition: NER
A process where a sentence or chunk of text is parsed through to find entities that can be put into categories.
(Person, Place, Organisation...)

spaCy has some pretrained models that we can refine to our needs.
Today we will be designing a new label and training a model to recognise it.

"""

import spacy
import random

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
         "eggs", "linguine", "pancakes", "parmigiana", "eggplant",
         "fettucine", "cream", "polenta", "risotto", "espresso",
         "pasta", "spaghetti", "fiorentina steak", "pecorino",
         "macaroni", "nutella", "amaro", "pistachio", "coca-cola",
         "wine", "pastiera", "watermelon", "cappuccino", "ice cream",
         "soup", "lemon", "chocolate", "pineapple", "nutella", "Tiramasu",
         "croissant", "soup", "bread"]

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
            
