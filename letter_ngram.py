
from collections import Counter
from typing import Tuple, List, Any, Set, Dict, Callable

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora
from nltk_model import *
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

import matplotlib.pyplot as plt

def hist(hh: List[float], title: str, align: str = 'mid',
         log: bool = False, block: bool = False):
  """
  Show a histgram with bars showing mean and standard deviations
  :param hh: the data to plot
  :param title: the plot title
  :param align: passed to pyplot.hist, q.v.
  :param log: passed to pyplot.hist, q.v.  If present will be added to title
  """
  hax=plt.subplots()[1] # Thanks to https://stackoverflow.com/a/7769497
  sdax=hax.twiny()
  hax.hist(hh,bins=30,color='lightblue',align=align,log=log)
  hax.set_title(title+(' (log plot)' if log else ''))
  ylim=hax.get_ylim()
  xlim=hax.get_xlim()
  m=np.mean(hh)
  sd=np.std(hh)
  sdd=[(i,m+(i*sd)) for i in range(int(xlim[0]-(m+1)),int(xlim[1]-(m-3)))]
  for s,v in sdd:
       sdax.plot([v,v],[0,ylim[0]+ylim[1]],'r' if v==m else 'pink')
  sdax.set_xlim(xlim)
  sdax.set_ylim(ylim)
  sdax.set_xticks([v for s,v in sdd])
  sdax.set_xticklabels([str(s) for s,v in sdd])
  plt.show(block=block)


def compute_accuracy(classifier, data: List[Tuple[List, str]]) -> float:
    """
    Computes accuracy (range 0 - 1) of a classifier.
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f: Callable[[str, str, str, str, str], List[Any]], data: List[Tuple[Tuple[str], str]])\
        -> List[Tuple[List[Any], str]]:
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


def get_annotated_tweets():
    """
    :rtype list(tuple(list(str), bool))
    :return: a list of tuples (tweet, a) where tweet is a tweet preprocessed by us,
    and a is True, if the tweet is in English, and False otherwise.
    """
    import ast
    with open("twitter/annotated_dev_tweets.txt") as f:
        return [ast.literal_eval(line) for line in f.readlines()]


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class: nltk.classify.api.ClassifierI, train_features: List[Tuple[List[Any], str]], **kwargs):
        """
        :param classifier_class: the kind of classifier we want to create an instance of.
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d: List[Any]) -> Dict[Any, int]:
        """
        :param d: list of features

        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d: List[Any]) -> str:
        """
        :param d: list of features

        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)


def train_LM(corpus: nltk.corpus.CorpusReader) -> LgramModel:
    """
    Build a bigram letter language model using LgramModel
    based on the lower-cased all-alpha subset of the entire corpus
    """
    
    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    corpus_tokens = [word.lower() for word in corpus.words() if word.isalpha()]
    lm = LgramModel(2, corpus_tokens,pad_left=True, pad_right=True)
    # Return the tokens and a smoothed (using the default estimator)
    # padded bigram letter language model
    return lm


def tweet_ent(file_name: str, bigram_model: LgramModel) -> List[Tuple[float, List[str]]]:
    """
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase
    """
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = [
        [word.lower() for word in tweet if word.isalpha()]
        for tweet in list_of_tweets]
    clean_tweets = [tweet for tweet in cleaned_list_of_tweets if len(tweet) >= 5]
    entropy_list = [(np.mean([bigram_model.entropy(word, perItem=True) for word in tweet]),tweet) for tweet in clean_tweets]
    sorted_solution = sorted(entropy_list, key = lambda x: x[0])
    return sorted_solution

def is_English(bigram_model: LgramModel, tweet: List[str]) -> bool:
    """
    Classify if the given tweet is written in English or not.
    """
    global ents
    ent = list(zip(*ents))[0]
    english_words = ent[:int(len(ent)*0.85)]
    cutoff = np.mean(english_words) + np.std(english_words)
    tweet_ent = np.mean([bigram_model.entropy(word,perItem=True) for word in tweet])
    return tweet_ent<cutoff



def stats():
    global ents, lm, top10_ents, bottom10_ents
    global lr_acc, logistic_regression_model, dev_features
    global dev_tweets_preds



    print('Building Brown news bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("Finding the cross entropy of each tweet")
    ents = tweet_ent(twitter_file_ids, lm)

    top10_ents = ents[:10]
    bottom10_ents = ents[-10:]


    print("Checking classfication accuracy")
    all_dev_ok = True
    dev_tweets_preds = []
    for tweet, gold_answer in get_annotated_tweets():
        prediction = is_English(lm, tweet)
        dev_tweets_preds.append(prediction)
        if prediction != gold_answer:
            all_dev_ok = False
            print("Missclassified", tweet)
    if all_dev_ok:
        print("All development examples correctly classified!")





if __name__ == "__main__":
    stats()
