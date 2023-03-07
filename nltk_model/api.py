# Natural Language Toolkit: API for Language Models
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Steven Bird <sb@csse.unimelb.edu.au>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT


# should this be a subclass of ConditionalProbDistI?

from typing import TypeVar, Tuple, List, Union, Sequence

Context = TypeVar('Context',Tuple[str, ...],List[str])
Ctxt = Union[List[str],Tuple[str, ...]]

class ModelI(object):
    """
    A processing interface for assigning a probability to the next word.
    """

    def __init__(self) -> None:
        """Create a new language model."""
        raise NotImplementedError()

    def prob(self, word: str, context: Ctxt) -> float:
        """Evaluate the probability of this word in this context."""
        raise NotImplementedError()

    def logprob(self, word: str, context: Ctxt) -> float:
        """Evaluate the (negative) log probability of this word in this context."""
        raise NotImplementedError()

    def choose_random_word(self, context: Ctxt) -> str:
        """Randomly select a word that is likely to appear in this context."""
        raise NotImplementedError()

    def generate(self, n: int) -> List[List[str]]:
        """Generate n words of text from the language model."""
        raise NotImplementedError()

    def entropy(self, text: Sequence[str]) -> float:
        """Evaluate the total entropy of a message with respect to the model.
        This is the sum of the log probability of each word in the message."""
        raise NotImplementedError()
