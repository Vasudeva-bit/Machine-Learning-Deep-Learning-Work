import pyjokes
import gradio as gr
import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('all')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import random
# import fastai

def similarity(input, joke):
    return cosine_similarity(input, joke)


def get_best(input):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    max_similarity = -1
    max_idx = 0
    jokes = pyjokes.get_jokes(language='en', category='all')
    jokes_embedding = model.encode(jokes)
    input_embedding = model.encode(input)
    for idx, joke_embedding in enumerate(jokes_embedding):
        sim = similarity(joke_embedding.reshape(-1, 1),
                         input_embedding.reshape(-1, 1))
        if(np.sum(sim) > np.sum(max_similarity)):
            max_idx = idx
            max_similarity = sim
    if(np.sum(max_similarity) != -1):
        return jokes[max_idx]+'😁🤣'
    else:
        return None


def generate_list(input):
    result = []
    n = len(input)
    for Len in range(2, n + 1):
        for i in range(n - Len + 1):
            j = i + Len - 1
            tem = ""
            for k in range(i, j + 1):
                tem += input[k]
            result.append(tem)
    return result


def pattern(input):
    response = input
    for substr in generate_list(input):
        try:
            syn = wn.synsets(substr)[1].hypernyms()[0].hyponyms()[
                0].hyponyms()[0].lemmas()[0].name()
        except:
            continue
        if(syn != None):
            response = response.replace(substr, syn.upper())
            break

    if(input == response):
        return None
    else:
        return response+'??😁🤣'

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict= dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def NLTK(input):
    f = open('corpus.txt', errors='strict')
    data = f.read()
    data = data.lower()
    data = data + input.lower()
    sent_tokens = nltk.sent_tokenize(data)
    return bot(sent_tokens)

def bot(sent_tokens):
  robo1_response = ''
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
  tfidf = TfidfVec.fit_transform(sent_tokens)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx = random.randint(0, len(vals.argsort()[0]))
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-1]
  if (req_tfidf == 0):
    robo1_response= robo1_response+"I could not answer this right now but you can contact the head of our dept (PUSPHA RAJ)." # add the dept recommendation engine and contact details
    return robo1_response
  else:
    robo1_response = robo1_response+sent_tokens[idx]
    return robo1_response

def generator(input=None):
    response = []
    if input:

        out1 = NLTK(input)
        if(out1):
            response.append(out1)

        out2 = pattern(input)
        if(out2):
            response.append(out2)

        out3 = get_best(input)
        if(out3):
            response.append(out3)

    else:
        out1 = NLTK("Hi, what's the matter")
        if(out1):
            response.append(out1)

        out2 = pyjokes.get_joke(language='en', category='all')
        if(out2):
            response.append(out2)

    return response  # think of doing this

iface = gr.Interface(fn=generator, inputs="text", outputs="text")
iface.launch()
