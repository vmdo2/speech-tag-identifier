import math
from collections import defaultdict, Counter
from math import log
import numpy as np
from queue import Queue
from queue import LifoQueue
import sys
import copy

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #raise NotImplementedError("You need to write this part!")
    seen = defaultdict(lambda: Counter())
    tags = Counter()
    sentences = []
    # train
    for i in range(len(train)):
        for word, tag in train[i]:
                seen[word][tag] += 1
                tags[tag] += 1
    # test
    for i in range(len(test)):
        insert = []
        for w in test[i]:
                if (w in seen):
                        insert.append((w, max(seen[w], key=seen[w].get)))
                else:
                        insert.append((w, max(tags, key=tags.get)))
        sentences.append(insert)
    return sentences

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    seen = defaultdict(lambda: Counter())
    tags = Counter()
    sentences = []
    num_pairs = 0

    # train
    for i in range(len(train)):
        for word, tag in train[i]:
                # skip if tag is start
                if tag == 'START':
                    continue
                seen[word][tag] += 1
                tags[tag] += 1
                num_pairs += 1

    # define epsilon for laplace smoothing
    k = 10**-9

    # intial probabilities:
    initial_prob = {}
    for tag in tags:
        initial_prob[tag] = 0
    for i in range(len(train)):
        initial_prob[train[i][1][1]] += 1
    total = sum(initial_prob.values())
    for key in initial_prob:
        initial_prob[key] = math.log((initial_prob[key] + k) / ((total) + k * (len(tags) + 1)))

    # transition probabilties:
    transition_prob = defaultdict(lambda: Counter())
    for tag in tags:
        for tag_ in tags:
            transition_prob[tag][tag_] = 0
    total = 0
    for i in range(len(train)):
        prev = "EMPTY"
        for word, tag in train[i]:
                if prev == "EMPTY":
                        prev = tag
                        continue
                transition_prob[prev][tag] += 1
                prev = tag
                total += 1

    for tag in transition_prob:
        total = sum(transition_prob[tag].values())
        for tag_ in transition_prob[tag]:
            transition_prob[tag][tag_] = math.log((transition_prob[tag][tag_] + k) / (total + (k * (len(tags) + 1))))
    
    # emission probabilities:
    emission_prob = defaultdict(lambda: Counter())
    total = 0
    for tag in tags:
        emission_prob[tag]["UNKNOWN"] = 0
    for i in range(len(train)):
            for word, tag in train[i]:
                # include ALL tags with word
                for t in tags:
                    if (word not in emission_prob[t]):
                        emission_prob[t][word] = 0
                emission_prob[tag][word] += 1
                total += 1

    summation = 0
    for key_1 in emission_prob:
        total = sum(emission_prob[key_1].values())
        for key_2 in emission_prob[key_1]:
            if key_2 == "UNKNOWN":
                emission_prob[key_1][key_2] = math.log(k / (total + k * (len(tags) + 1)))
            else:
                emission_prob[key_1][key_2] = math.log((emission_prob[key_1][key_2] + k) / (total + k * (len(tags) + 1)))
            summation += emission_prob[key_1][key_2]

    result = []
    for i in range(len(test)):
        # initialize trellis for each sentence
        trellis = []
        for j in range(len(tags)):
            insert = []
            # do len(test[i])-1 to begin at first non-'START' tag
            for k in range(len(test[i])-1):
                insert.append(0)
            trellis.append(insert)
        # backtrack 2d list has same dimensions
        pointers = copy.deepcopy(trellis)
        index = 0
        for tag in tags:
            # check if word exists in emission
            if test[i][1] in emission_prob[tag]:
                trellis[index][0] = initial_prob[tag]+emission_prob[tag][test[i][1]]
            else:
                trellis[index][0] = initial_prob[tag]+emission_prob[tag]["UKNOWN"]
            index += 1
        for j in range(2, len(test[i])):
            for index, key_1 in enumerate(tags):
                formula = 0
                if (test[i][j] in emission_prob[key_1]):
                    formula = emission_prob[key_1][test[i][j]]
                else:
                    formula = emission_prob[key_1]["UNKNOWN"]
                best_path_percentage = -1*((2**63-1)-1)
                best_k = 0
                best_tag = 'EMPTY'
                for k, key_2 in enumerate(tags):
                    # get best k value (argmax of word/tag)
                    if (trellis[k][j-2]+transition_prob[key_2][key_1]+formula > best_path_percentage):
                        best_path_percentage = trellis[k][j-2]+transition_prob[key_2][key_1]+formula
                        best_k = k 
                        best_tag = key_2
                    if (k >= len(tags)):
                        break
                trellis[index][j-1] = trellis[best_k][j-2]+transition_prob[best_tag][key_1]+formula
                # store k value to backtrack later
                pointers[index][j-1] = best_k
        # start back track
        best_path = []
        best_path_percentage = -1*((2**63-1)-1)
        k_best = 0
        for k, tag in enumerate(tags):
            if (trellis[k][len(test[i])-2] > best_path_percentage):
                best_path_percentage = trellis[k][len(test[i])-2]
                k_best = k
        for j in range(len(test[i])-1, 0, -1,):
            for k, tag in enumerate(tags):
                if (k == k_best):
                    best_path.append((test[i][j], tag))
                    break
            k_best = pointers[k_best][j-1]

        # manually add start pair
        best_path.append(('START', 'START'))
        best_path.reverse()
        result.append(best_path)
    return result