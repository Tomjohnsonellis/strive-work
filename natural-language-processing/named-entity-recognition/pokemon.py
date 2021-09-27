# After going through "training_a_ner.py", I decided to do it again
# but this time with a more fun dataset.

# I will need the same utilities
import random
import spacy
import random
from spacy.util import minibatch
from spacy.training import Example

# First off, I will build the dataset.
# I'll need:
# A list of all the entities I want to recognise
# A load of sentences containing those entities
# The locations of the entities in those sentences.

# The entities
def get_entities():
    with open("natural-language-processing/named-entity-recognition/data/pokemon.txt") as file:
        entities_to_learn = file.readlines()

        # Convert to lowercase
        entities_to_learn = [e.lower() for e in entities_to_learn]
        # Remove newline characters
        entities_to_learn = [e.replace("\n","") for e in entities_to_learn]


        return entities_to_learn

# Generate Sentences
# I'll generate some simple sentences to train on
def generate_sentences(entities, extra_data=300):

    training_sentences = []

    base_sentences = [
        "x is my favourite!",
        "I think x is cool.",
        "What moves does x know?",
        "Who uses x lmao",
        "Where can I find a(n) x"
    ]

    complex_bases = [
        "x is way better than y",
        "x and y are my favourites"
        "my x lost to a(n) y"
    ]

    # Generate a simple sentence for each entity
    for entity in entities:
        # Pick a random sentence from above
        i = random.randint(0, len(base_sentences)-1)

        # While we are here, we can work out the entity's location in the sentence,
        # This is needed later for training
        starting_index = base_sentences[i].index("x")
        ending_index = starting_index + len(entity)
        # Create a sentence
        text = base_sentences[i].replace("x",entity)
        # Store the needed info
        location_info = (starting_index, ending_index, "POKEMON")
        # Store it all in our training dataset
        training_sentences.append( (text, {"entities":location_info}))

    # Extension: Create more complex sentences


    return training_sentences




if __name__ == "__main__":
    pokemon = get_entities()
    sentences = generate_sentences(pokemon)
    print(sentences)