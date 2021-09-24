"""
Goal of today is to make a simple chatbot therapist.
We have trained a simple model on a text file of words and responses.
Now we need to take the user's input, feed it through that model,
and use the model's response to determine the next action.

E.g.
User says "Hello"
Give "Hello" to model
model thinks "Hello" is a "Greeting"
model responds accordingly
"""


import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



def print_intro(name) -> None:

    with open("natural-language-processing/chatbot/logo.txt") as f:
        print("".join([line for line in f]))
    print()
    print("-"*100)
    print(f"Today's friend is called: {name}")
    print("-"*100)
    return None


# # Setup
def setup():
    # In case there's a GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load our trained model
    model_path = "natural-language-processing/chatbot/trained.pth"
    model = torch.load(model_path)
    # Get some information about it
    model_info = {
        "input_size":model["input_size"],
        "hidden_size":model["hidden_size"],
        "output_size":model["output_size"],
        "all_words":model['all_words'],
        "tags":model['tags'],
        "model_state":model["model_state"],
    }


    # This was problematic, possibly worth persuing a solution
    # # We'll also add the possible intents for later
    # with open('natural-language-processing/chatbot/intents.json', 'r') as json_data:
    #     intents = json.load(json_data)
    #     model_info.update({"intents":intents})

    # Instantiate a model for the chatbot
    chat_model = NeuralNet(model_info["input_size"], model_info["hidden_size"], model_info["output_size"]).to(device)
    chat_model.load_state_dict(model_info["model_state"])
    # We've already trained it, so switch it to evaluation mode
    chat_model.eval()

    return chat_model, model_info

def name_bot():
    potential_names = [
        "Larry",
        "Edgar",
        "Mr. Owl",
        "Ms. Molly",
        "Brenda",
        "THE VOID"
    ]

    i = random.randint(0, len(potential_names)-1)
    return potential_names[i]

def talk_to_bot(chatbot, chatinfo, bot_name, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    user_statement = input("Please type your message: ")
    if user_statement == "exit":
        print("Exiting chat session...")
        return False
    
    # Tokenize what the user has typed so we can give it to the model
    sentence_to_evaluate = tokenize(user_statement)
    X = bag_of_words(sentence_to_evaluate, chatinfo["all_words"])
    # Reshape it to a 1d vector as opposed to a bunch of individual values
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    model_output = chatbot(X)
    # Get the most confident prediction
    _, prediction = torch.max(model_output, dim=1)
    # Get the corresponding human readable version
    tag = chatinfo["tags"][prediction.item()]

    if verbose:
        print(f"Model predicts: {tag}")

    # Calculate the confidence
    probabilities = torch.softmax(model_output, dim=1)
    confidence = probabilities[0][prediction.item()]
    if verbose:
        print(confidence)

   

    with open('natural-language-processing/chatbot/intents.json', 'r') as json_data:
        response_data = json.load(json_data)


    # If the model is quite confident, give a response
    if confidence.item() > 0.8:
        # Check the tags
        for intent in response_data["intents"]:

            if tag == intent["tag"]:
                # Print a random appropriate response
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    # If we aren't sure
    else:
        print(f"{bot_name}: I do not understand. Try to be more specific")



    return True

if __name__ == "__main__":
    chatbot, chatinfo = setup()
    bot_name = name_bot()
    print_intro(bot_name)
    print("-[ You start the conversation. ]-")
    continue_conversation = True
    while continue_conversation:
        continue_conversation = talk_to_bot(chatbot, chatinfo, bot_name)

    
