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
        pos = (starting_index, ending_index, "POKEMON")
        # This looks a bit odd, but we want to have a list of all entities, in this case it will always be one element,
        # But for more complex sentences we may want more than one
        locs = []
        locs.append(pos)
        # Store it all in our training dataset
        element = (text, {"entities":locs})
        training_sentences.append(element)

    # Extension: Create more complex sentences


    return training_sentences

# Now we have a dataset to work with.
# We can now begin update a model.
def train_model(nlp_model, training_data, epochs=10, save=False):
    # Load a model then pass it here for updating
    # Add our new tag
    pokemon_ner = nlp_model.get_pipe("ner")
    pokemon_ner.add_label("FOOD")
    pokemon_ner.add_label("POKEMON")

    # Disable everything we don't need for fast training
    things_to_affect = ["ner", "trf_wordpiecer","trf_tok2vec"]
    unaffected = [pipe for pipe in nlp_model.pipe_names if pipe not in things_to_affect]
    # Using only the things we need...
    with nlp_model.disable_pipes(*unaffected):
        for iteration in range(epochs):
            # Shuffle the data, to help prevent overfitting
            random.shuffle(training_data)
            losses = {}

            # Batch up the data
            batches = minibatch(training_data, size=32)
            # Work through each batch
            for index, batch in enumerate(batches):
                for text, entities in batch:
                    # Make a doc out of the text
                    doc = nlp_model.make_doc(text)
                    # Create a spaCy "Example", which is two Docs, one is the correctly labelled info, the other is the pipeline's predictions
                    example = Example.from_dict(doc, entities)
                    # Update our nlp model with the info
                    nlp_model.update([example], losses=losses, drop=0)
                print(f"Iteration: {iteration} | Batch: {index} | Loss: {losses}")
        
    if save:
        nlp_model.to_disk("natural-language-processing/named-entity-recognition/data/poke.nlp")
        print("Saved!")
    
    return nlp_model


def test_model(model):
    test_one = model("My charizard loves soup")
    test_two = model("Don't give pasta to rattata")
    test_three = model("I gave slowbro a cappuccino")
    tests = [test_one, test_two, test_three]
    for doc in tests:
        print("Entities: ", [(entity.text, entity.label_) for entity in doc.ents])
    return


def get_food_training():
    words = ["ketchup", "pasta", "carrot", "pizza",
         "garlic", "tomato sauce", "basil", "carbonara",
         "eggs", "cheek fat", "pancakes", "parmigiana", "eggplant",
         "fettucine", "heavy cream", "polenta", "risotto", "espresso",
         "arrosticini", "spaghetti", "fiorentina steak", "pecorino",
         "maccherone", "nutella", "amaro", "pistachio", "coca-cola",
         "wine", "pastiera", "watermelon", "cappuccino", "ice cream",
         "soup", "lemon", "chocolate", "pineapple"]

    with open("natural-language-processing/named-entity-recognition/data/food.txt") as foodfile:
       food_dataset = foodfile.readlines()
    food_dataset = [sentence.lower() for sentence in food_dataset]
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

    return train_data


def combine_datasets(food, pokemon):
    big_dataset = []
    for x in food:
        big_dataset.append(x)
    for x in pokemon:
        big_dataset.append(x)

    return big_dataset

if __name__ == "__main__":
    pokemon = get_entities()
    poke_sentences = generate_sentences(pokemon)
    food_sentences = get_food_training()
    # print(food_sentences)
    # print("-"*50)
    combined_data = combine_datasets(food_sentences, poke_sentences)
    
    food_nlp = spacy.load("natural-language-processing/named-entity-recognition/data/trained.nlp")
    test_model(food_nlp)
    food_and_poke = train_model(food_nlp, combined_data, save=True)

    test_model(food_and_poke)