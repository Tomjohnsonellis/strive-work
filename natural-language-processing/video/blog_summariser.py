"""
This is a program that will attempt to shorten a given blog post 
with the use of huggingface's text summary model.

Many thanks to: https://github.com/nicknochnack/Longform-Summarization-with-Hugging-Face/blob/main/LongSummarization.ipynb
for the tutorial and accompanying video: https://www.youtube.com/watch?v=JctmnczWg0U
"""


from re import split
from transformers import pipeline # Huggingface
from bs4 import BeautifulSoup as bs
import requests


def get_article_text(blog_url):

    r = requests.get(blog_url)
    soup = bs(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    text = [result.text for result in results]
    blog_text = ' '.join(text)

    return blog_text

# There are limits on how much text we can put through a pipeline, so we will split it into chunks
def split_text(all_text, max_length=750):
    # Full stops are one way of splitting text.
    text_to_split = all_text.replace(". ", ".<eos> ") # eos meaning "End of Section"
    sentences = text_to_split.split("<eos>")
    # We want to make chunks of text that are smaller than the max_length
    current_chunk = 0
    chunks = [""]

    for sentence in sentences:
        if len(chunks[current_chunk]) + len(sentence) < max_length:
            chunks[current_chunk] += (sentence)
        else:
            chunks.append("")
            current_chunk += 1
            chunks[current_chunk] += (sentence)

    return chunks

def get_stats(text_chunks, summary_results, verbose=False):
    original = 0
    summary = 0
    for t in text_chunks:
        original += len(t)
    
    for s in summary_results:
        summary += len( s['summary_text'] )
    
    diff = abs(original - summary)

    stats = {"original_length":original, "summary_length":summary, "savings":diff}

    if verbose:
        print("-"*50)
        for k in stats:
            print(f"{k} | {stats[k]}")
        print("-"*50)


    return stats


def print_summary(summary_results):

    print("="*50)
    for short_sentence in summary_results:
        print(short_sentence["summary_text"])
    print("="*50)
    

    
    return

def give_me_summary_and_stats(url):
    summarizer = pipeline("summarization")
    blog_text = get_article_text(url)
    chunks = split_text(blog_text)
    res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    stats = get_stats(chunks, res)
    return res, stats


if __name__ == "__main__":
    summarizer = pipeline("summarization")

    blog_url = "https://estherilori.medium.com/how-data-science-became-a-lifestyle-13d5dae35883"
    blog_text = get_article_text(blog_url)
    # print(blog_text)
    chunks = split_text(blog_text)

    res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    stats = get_stats(chunks, res, True)



    print_summary(res)

    
