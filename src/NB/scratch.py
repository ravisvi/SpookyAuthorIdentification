# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:26:34 2018

@author: OH YEA
"""

import nltk

from nltk import FreqDist

otherSent = "This is not there for you is."
allWords = nltk.word_tokenize(otherSent)

print(allWords)

otherWords = FreqDist(allWords)

print(otherWords[0])