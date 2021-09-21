# These are a few pre-processing functions in order to understand what it happening
# at the very basic level, no fancy tecniques, just base python lists and dictionaries
import numpy as np

def remove_symbols(file_path:str) -> None:
    """
    For NLP, punctuation marks are likely to be of more harm than good,
    So we will just remove them.
    """
    symbols = "!\"£$%^&*()\',./#;:[]"

    # Open a file and remove the above symbols
    with open(file_path, "r") as base_file:
        lines = base_file.readlines()
        for index, _ in enumerate(lines):
            for symbol in symbols:
                lines[index] = lines[index].replace(symbol,"")
    
    # Save it as "<file>-nosym.txt"
    with open(file_path[:-4] + "-nosym.txt", "w") as processed_file:
        processed_file.writelines(lines)
    return

def get_sentences(file_path:str) -> list[list[str]]:
    """
    A function to grab each line as a list of words
    """
    with open(file_path, "r") as corpus:
        lines = corpus.readlines()
        sentences = [line.upper().split() for line in lines]

    return sentences

def remove_blanks(sentences:list[list[str]]) -> list[list[str]]:
    """
    Gets rid of any empty lists
    """
    blanks = True
    while blanks:
        blanks = False
        for index, sentence_to_check in enumerate(sentences):
            if sentence_to_check == []:
                sentences.pop(index)
                blanks = True
            
    return sentences


def create_dictionary(sentences):
    """
    For the corpus we are working with, give us some information about it.
    We're finding the first occurence of each unique word and storing
    both the word and it's position, we will be able to use these to
    turn sentences into vectors
    """
    vocabulary = []

    # Go through the corpus and add anything new
    for sentence in sentences:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2index = { word:index for (index, word) in enumerate(vocabulary)}
    index2word = { index:word for (index, word) in enumerate(vocabulary)}

    # We will need the length of the vocabulary as that's the tensor size for an embedding
    return word2index, index2word, len(vocabulary)


def get_pairs(sentences, word_to_index:dict, context_range=1):
    """
    In order to get some sort of context, we're going to find pairs of words
    This function is like pointing to the words nearby and then doing something
    with them if it's valid
    """
    pairs = []
    cr = context_range
    for sentence in sentences:
        # We now make a numerical representation of the sentence
        tokens = [ word_to_index[word] for word in sentence]

        # Next up, we want to make a word pair out of some word and its neighbour
        for focus_point in range(len(tokens)):
            # For each possible word in our range
            for context_point in range(-cr, cr+1):
                current_point = focus_point + context_point
                
                # With this method, we do not want to progress if
                # The current point doesn't point to anything
                # or it is pointing to the word we're working on.
                if current_point < 0 or current_point >= len(tokens) or current_point == focus_point:
                    # If it's not something we're interested in, skip it
                    continue
                else:
                    # We have a pair, add it to the list
                    pairs.append( (tokens[focus_point], tokens[current_point]))

    return pairs

        

def pair_pipeline(file_path, complexity=1):
    remove_symbols(file_path)
    raw_sentences = get_sentences(file_path[:-4] + "-nosym.txt")
    no_blanks = remove_blanks(raw_sentences)
    word2index, _, vocabulary_size = create_dictionary(no_blanks)
    pairs = get_pairs(no_blanks, word2index, complexity)
    return np.array(pairs), vocabulary_size


pairs, words = pair_pipeline("natural-language-processing/data/poems.txt")
print(pairs)
print(words)



# some_sentences = get_sentences("natural-language-processing/data/poems-nosym.txt")
# # print(len(some_sentences))
# no_blanks = remove_blanks(some_sentences)
# # print(len(no_blanks))
# w2i, i2w, word_count = create_dictionary(no_blanks)
# print(word_count)
# print(w2i["POEMS"])
# print(i2w[0])
# pairs = get_pairs(no_blanks, w2i)
# print(len(pairs))


# print(no_blanks)

# if __name__ == "__main__":
#     remove_symbols("natural-language-processing/data/poems.txt")