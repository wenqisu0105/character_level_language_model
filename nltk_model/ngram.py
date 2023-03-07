# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2009 NLTK Project
# Author: Steven Bird <sb@csse.unimelb.edu.au>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

import random, collections.abc
from itertools import chain
from math import log

from nltk.probability import (ConditionalProbDist, ConditionalFreqDist, ProbDistI,
                              MLEProbDist, FreqDist, WittenBellProbDist)

from nltk.util import ngrams as ingrams

from typing import Tuple, List, Iterable, Any, Set, Dict, Callable
from typing import Union, Optional, Sized, cast, Sequence, TextIO

AlphaDict = Dict[Tuple[str, ...],float]

try:
    from api import *
except ImportError:
    from .api import *

def isclose(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    '''
    Courtesy of http://stackoverflow.com/a/33024979

    Test if two numbers are, as it were, close enough.

    Note that float subsumes int for type-checking purposes,
    so ints are OK, so e.g. isclose(1,0.999999999999) -> True.
    '''
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def discount(self: WittenBellProbDist) -> float:
    return float(self._N)/float(self._N + self._T)

def check(self: WittenBellProbDist) -> None:
    totProb=sum(self.prob(sample) for sample in self.samples())
    assert isclose(self.discount(),totProb),\
           "discount %s != totProb %s"%(self.discount(),totProb)

WittenBellProbDist.discount = discount
WittenBellProbDist.check = check

def _estimator(fdist: FreqDist, bins: int) -> WittenBellProbDist:
    """
    Default estimator function using WB.
    """
    res=WittenBellProbDist(fdist,fdist.B()+1)
    res.check()
    return res

class NgramModel(ModelI):
    """
    A processing interface for assigning a probability to the next word.
    """

    def __init__(self, n: int, train: Union[Iterable[str],Iterable[Iterable[str]]],
                 pad_left: bool = False, pad_right: bool = False,
                 estimator: Optional[Callable[[FreqDist,int],ProbDistI]] = None,
                 *estimator_args, **estimator_kwargs) -> None:
        """
        Creates an ngram language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training.

        :param n: the order of the language model (ngram size)
        :param train: the training text
        :param estimator: a function for generating a probability distribution.
                          Defaults to lambda fdist, bins: MLEProbDist(fdist)
        :param pad_left: whether to pad the left of each sentence with an (n-1)-gram of <s>
        :param pad_right: whether to pad the right of each sentence with </s>
        :param estimator_args: Extra arguments for estimator.
            These arguments are usually used to specify extra
            properties for the probability distributions of individual
            conditions, such as the number of bins they contain.
            Note: For backward-compatibility, if no arguments are specified, the
            number of bins in the underlying ConditionalFreqDist are passed to
            the estimator as an argument.
        :param estimator_kwargs: Extra keyword arguments for the estimator
        """

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert isinstance(pad_left, bool)
        assert isinstance(pad_right, bool)

        # make sure n is greater than zero, otherwise print it
        assert (n > 0), n

        # For explicitness save the check whether this is a unigram model
        self.is_unigram_model = (n == 1)
        # save the ngram order number
        self._n = n
        # save left and right padding
        self._lpad = ('<s>',) * (n - 1) if pad_left else ()
        # Need _rpad even for unigrams or padded entropy will give
        #  wrong answer because '</s>' will be treated as unseen...
        self._rpad = ('</s>',) if pad_right else ()
        self._padLen = len(self._lpad)+len(self._rpad)

        self._N=0
        delta = 1+self._padLen-n        # len(sent)+delta == ngrams in sent
        self._delta = delta

        if estimator == None:
            assert (estimator_args==()) and (estimator_kwargs=={}),\
                   "estimator_args (%s) or _kwargs supplied (%s), but no estimator"%(estimator_args,estimator_kwargs)
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # Given backoff, a generator isn't acceptable
        if not isinstance(train,List):
          train = list(train)
        self._W = len(train)
        # Coerce to list of list -- note that this means to train charGrams,
        #  requires exploding the words ahead of time 
        if train != None:
            if isinstance(train[0],str):
                train = [train]
                self._W=1
            elif not isinstance(train[0],collections.abc.Sequence):
                # if you mix strings and generators, you have only yourself
                #  to blame!
                for i in range(len(train)):
                    train[i]=list(train[i])

        if n == 1:
            if pad_right:
                sents=(chain(s,self._rpad) for s in train)
            else:
                sents=train
            fd=FreqDist()
            for s in sents:
                fd.update(s)
            if not estimator_args and not estimator_kwargs:
                self._model = estimator(fd,fd.B())
            else:
                self._model = estimator(fd,fd.B(),
                                        *estimator_args, **estimator_kwargs)
            self._N=fd.N()
        else:
            cfd = ConditionalFreqDist()
            self._ngrams = set()

            for sent in train:
                self._N+=len(sent)+delta
                for ngram in ingrams(chain(self._lpad, sent, self._rpad), n):
                    self._ngrams.add(ngram)
                    context = tuple(ngram[:-1])
                    token = ngram[-1]
                    cfd[context][token]+=1
            if not estimator_args and not estimator_kwargs:
                self._model = ConditionalProbDist(cfd, estimator, len(cfd))
            else:
                self._model = ConditionalProbDist(cfd, estimator, *estimator_args, **estimator_kwargs)

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            self._backoff = NgramModel(n-1, train,
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args,
                                        **estimator_kwargs)

            # Code below here in this method, and the _words_following and _alpha method, are from
            # http://www.nltk.org/_modules/nltk/model/ngram.html "Last updated on Feb 26, 2015"
            self._backoff_alphas = dict()
            # For each condition (or context)
            for ctxt in cfd.conditions():
                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0

                # this is the subset of words that we OBSERVED following
                # this context.
                # i.e. Count(word | context) > 0
                for word in self._words_following(ctxt, cfd):
                    total_observed_pr += self.prob(word, ctxt)
                    # we also need the total (n-1)-gram probability of
                    # words observed in this n-gram context
                    backoff_total_pr += self._backoff.prob(word, backoff_ctxt)
                if isclose(total_observed_pr,1.0):
                    total_observed_pr=1.0
                else:
                    assert 0.0 <= total_observed_pr <= 1.0,\
                           "sum of probs for %s out of bounds: %.10g"%(ctxt,total_observed_pr)
                # beta is the remaining probability weight after we factor out
                # the probability of observed words.
                # As a sanity check, both total_observed_pr and backoff_total_pr
                # must be GE 0, since probabilities are never negative
                beta = 1.0 - total_observed_pr

                if beta!=0.0:
                    assert (0.0 <= backoff_total_pr < 1.0), \
                           "sum of backoff probs for %s out of bounds: %s"%(ctxt,backoff_total_pr)
                    alpha_ctxt = beta / (1.0 - backoff_total_pr)
                else:
                    assert ((0.0 <= backoff_total_pr < 1.0) or
                            isclose(1.0,backoff_total_pr)), \
                           "sum of backoff probs for %s out of bounds: %s"%(ctxt,backoff_total_pr)
                    alpha_ctxt = 0.0

                self._backoff_alphas[ctxt] = alpha_ctxt

    def _words_following(self, context: Tuple[str], cond_freq_dist: ConditionalFreqDist) -> Sequence[str]:
        return cast(Sequence[str], cond_freq_dist[context].keys())

    def prob(self, word: str, context: Ctxt,
             verbose: bool = False) -> float:
        """
        Evaluate the probability of this word in this context.
        Will use Katz Backoff if the underlying distribution supports that.

        :param word: the word to get the probability of
        :param context: the context the word is in
        """
        res: float
        assert(isinstance(word,str))
        context = tuple(context)
        if self.is_unigram_model:
            if not(self._model.SUM_TO_ONE):
                # Smoothing models should do the right thing for unigrams
                #  even if they're 'absent'
                res = self._model.prob(word)
            else:
                try:
                    res = self._model.prob(word)
                except:
                    raise RuntimeError("No probability mass assigned"
                                       "to unigram %s" % (word))
        elif context + (word,) in self._ngrams:
            res = self[context].prob(word)
        else:
            alpha=self._alpha(context)
            if alpha>0.0:
                if verbose:
                    print("backing off for %s"%(context+(word,),))
                res = alpha * self._backoff.prob(word, context[1:],verbose)
            else:
                if verbose:
                    print('no backoff for "%s" as model doesn\'t do any smoothing so prob=0.0'%word)
                res = alpha
        if verbose:
            print("p(%s|%s) = [%s-gram] %7f"%(word,context,self._n,res))
        return res
        
    def _alpha(self, context: Ctxt,
               verbose: bool = False) -> float:
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self.is_unigram_model, error_message

        if context in self._backoff_alphas:
            res = self._backoff_alphas[context]
        else:
            res = 1
        if verbose:
            print(" alpha: %s = %s"%(context,res))
        return res


    def logprob(self, word: str, context: Ctxt, verbose: bool = False) -> float:
        """
        Compute the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :param context: the context the word is in
        """

        return -log(self.prob(word, context,verbose), 2)

    @property
    def ngrams(self) -> Set[Tuple[str]]:
        return self._ngrams

    @property
    def backoff(self) -> 'NgramModel':
        return self._backoff

    @property
    def model(self) -> Union[ProbDistI,ConditionalProbDist]:
        return self._model

    # NB, this will always start with same word since model
    # is trained on a single text
    def generate(self, num_words: int,
                 context: Ctxt = ()) -> List[List[str]]:
        '''
        Generate random text based on the language model.

        :param num_words: number of words to generate
        :param context: initial words in generated string
        '''

        orig = list(context)
        res=[]
        text = list(orig) # take a copy
        for i in range(num_words):
            one=self._generate_one(text)
            text.append(one)
            if one=='</s>' or i==num_words-1:
                if self._lpad!=():
                    res.append(list(self._lpad)[:(len(self._lpad)+len(context))-(self._n-2)]+text)
                else:
                    res.append(text)
                text=list(orig)
        return res

    def _generate_one(self, context: Ctxt) -> str:
        context = (self._lpad + tuple(context))[-self._n+1:]
        # print "Context (%d): <%s>" % (self._n, ','.join(context))
        if context in self:
            return cast(str,self[context].generate())
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return cast(str,self._model.max())

    def entropy(self, text: Sequence[str], verbose: bool = False,
                perItem: bool = False) -> float:
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given evaluation text.
        This is either the sum or the average (see perItem) of the
        cost (negative log probability) of each item in the text.

        :param text: items to use for evaluation
        :param perItem: normalise for text length if True
        """
        # This version takes account of padding for greater accuracy
        #  if the model was built with padding
        # Note that if input is a string, it will be exploded into characters 
        e = 0.0
        for ngram in ingrams(chain(self._lpad, text, self._rpad), self._n):
            context = tuple(ngram[:-1])
            token = ngram[-1]
            cost=self.logprob(token, context, verbose)  # _negative_
                                                        # log2 prob == cost!
            e += cost
        if perItem:
            return e/(len(text)+self._delta)
        else:
            return e

    def perplexity(self, text: Sequence[str], verbose: bool = False) -> float:
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.

        :param text: words to calculate perplexity of
        """

        return pow(2.0, self.entropy(text, verbose=verbose, perItem=True))

    def dump(self, file: TextIO, logBase: Optional[float] = None,
             precision: int = 7) -> None:
        """Dump this model in SRILM/ARPA/Doug Paul format

        Use logBase=10 and the default precision to get something comparable
        to SRILM ngram-model -lm output
        @param file to dump to
        @param logBase If not None, output logs to the specified base"""
        file.write('\n\\data\\\n')
        self._writeLens(file)
        self._writeModels(file,logBase,precision,None)
        file.write('\\end\\\n')

    def _writeLens(self,file: TextIO) -> None:
        if self._n>1:
            self._backoff._writeLens(file)
            file.write('ngram %s=%s\n'%(self._n,
                                        sum(len(self._model[c].samples())\
                                            for c in self._model.keys())))
        else:
            file.write('ngram 1=%s\n'%len(self._model.samples()))
            

    def _writeModels(self, file: TextIO, logBase: Optional[float],
                     precision: int, alphas: Optional[AlphaDict]) -> None:
        if self._n>1:
            self._backoff._writeModels(file, logBase, precision, self._backoff_alphas)
        file.write('\n\\%s-grams:\n'%self._n)
        if self._n==1:
            self._writeProbs(self._model, file, logBase, precision, (), alphas)
        else:
            for c in sorted(self._model.conditions()):
                self._writeProbs(self._model[c], file, logBase, precision, c, alphas)

    def _writeProbs(self, pd: ProbDistI, file: TextIO, logBase: Optional[float],
                    precision: int, ctxt: Tuple[str, ...],
                    alphas: Optional[AlphaDict]) -> None:
        if self._n==1:
            for k in sorted(chain(pd.samples(),['<unk>','<s>'])):
                if k=='<s>':
                    file.write('-99')
                elif k=='<unk>':
                    _writeProb(file, logBase, precision, 1-pd.discount()) 
                else:
                    _writeProb(file, logBase, precision, pd.prob(k))
                file.write('\t%s'%k)
                if k not in ('<s>','<unk>'):
                    try:
                        bv = alphas[ctxt+(k,)]
                        file.write('\t')
                        _writeProb(file, logBase, precision, bv)
                    except (TypeError, KeyError):
                        pass
                file.write('\n')
        else:
            ctxtString=' '.join(ctxt)
            for k in sorted(pd.samples()):
                _writeProb(file, logBase, precision, pd.prob(k))
                file.write('\t%s %s'%(ctxtString,k))
                try:
                    bv = alphas[ctxt+(k,)]
                    file.write('\t')
                    _writeProb(file, logBase, precision, bv)
                except (TypeError, KeyError):
                    pass
                file.write('\n')

    def __contains__(self, item: Sequence[str]) -> bool:
        item=tuple(item)
        try:
            return item in self._model
        except:
            try:
                # hack if model is an MLEProbDist, more efficient
                return item in self._model._freqdist
            except:
                return item in self._model.samples()

    def __getitem__(self, item: Sequence[str]) -> ProbDistI:
        return cast(ProbDistI,self._model[tuple(item)])

    def __repr__(self) -> str:
        return '<NgramModel with %d %d-grams>' % (self._N, self._n)

def _writeProb(file: TextIO, logBase: Optional[float],
               precision: int, p: float) -> None:
    file.write('%.*g'%(precision, 
                       p if logBase == None else log(p,cast(float,logBase))))


class LgramModel(NgramModel):
    def __init__(self, n: int, train: Iterable[str],
                 pad_left: bool = False, pad_right: bool = False,
                 estimator: Optional[Callable[[FreqDist,int],ProbDistI]] = None,
                 *estimator_args, **estimator_kwargs) -> None:
        """
        NgramModel (q.v.) slightly tweaked to produce char-grams,
        not word-grams, with a WittenBell default estimator

        :param train: List of strings, which will be converted to list of lists of characters, but more efficiently

        For other parameters, see NgramModel.__init__
        """
        if estimator == None:
            assert (not(estimator_args)) and (not(estimator_kwargs)),\
                   "estimator_args (%s) or _kwargs (%s) supplied, but no estimator"%(estimator_args,estimator_kwargs)
            estimator=_estimator
        super(LgramModel,self).__init__(n,
                                        (iter(word) for word in train),
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args, **estimator_kwargs)

def teardown_module() -> None:
    from nltk.corpus import brown
    brown._unload()

from nltk.probability import LidstoneProbDist

def demo(estimator_class: ProbDistI = LidstoneProbDist) -> NgramModel:
    from nltk.corpus import brown
    estimator = lambda fdist, bins: estimator_class(fdist, 0.2, bins+1)
    lm = NgramModel(3, brown.sents(categories='news'), estimator=estimator,
                    pad_left=True, pad_right=True)
    print("Built %s using %s for underlying probability distributions"%(lm,estimator_class))
    txt="There is no such thing as a free lunch ."
    print("Computing average per-token entropy for \"%s\", showing the computation:"%txt)
    e=lm.entropy(txt.split(), verbose=True, perItem=True)
    print("Per-token average: %.2f"%e)
    text = lm.generate(100)
    import textwrap
    print("--------\nA randomly generated 100-token sequence:")
    for sent in text:
        print('\n'.join(textwrap.wrap(' '.join(sent))))
    return lm

if __name__ == '__main__':
    demo()



