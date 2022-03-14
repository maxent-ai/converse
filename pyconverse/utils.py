import re
import json
import spacy
import string
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


@lru_cache
def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model


@lru_cache
def load_spacy():
    return spacy.load('en_core_web_sm')


def load_zeroshot_model(model_name="facebook/bart-large-mnli"):
    classifier = pipeline("zero-shot-classification", model=model_name)
    return classifier


def load_summarization_model(model="knkarthick/MEETING_SUMMARY"):
    model = pipeline("summarization", model=model)
    return model


def remove_punct(text):
    punct_list = re.compile('[%s]' % re.escape(string.punctuation))
    text = re.sub(punct_list, ' ', text)
    text = re.sub("  +", ' ', text)
    return text


def json_reader(path):
    data = None
    with open(path, 'r') as f:
        data = json.load(f)
    return data
