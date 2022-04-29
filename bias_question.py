import pandas as pd
import re
from typing import Iterator, Iterable, Tuple, Text, Union
import math
import numpy as np
from scipy.sparse import spmatrix

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

import csv
import statistics as st
import os

NDArray = Union[np.ndarray, spmatrix]

np.random.seed(1)

def lookup_tables(filename : str):
    '''
    

    returns people_dict, emotions_dict:
        {Name1: [gender, race], ...}
        {general_emotion1: [emotion_word1, emotion_word2, ...], ...}
    '''
    csvFile = pd.read_csv(filename)
    people = {}
    condense_list = []
    for i in range(len(csvFile)):
        condense_list.append('~'.join([str(csvFile["Person"][i]), str(csvFile["Gender"][i]), str(csvFile["Race"][i])]))
    condense_list = list(set(condense_list))
    for item in condense_list:
        person = item.split('~')
        if person[2] == 'nan':
            person[2] = re.sub('nan', 'NA', person[2])
        people[person[0]] = [person[1], person[2]]
    pout = {}
    for p in sorted(people):
        pout[p] = people[p]
    
    emotions = {}
    condense_list = []
    for i in range(len(csvFile)):
        if type(csvFile["Emotion"][i]) == str:
            condense_list.append('~'.join([str(csvFile["Emotion"][i]), str(csvFile["Emotion word"][i])]))
    condense_list = list(set(condense_list))
    for item in condense_list:
        emotion = item.split('~')
        if emotion[0] in emotions:
            emotions[emotion[0]].append(emotion[1])
        else:
            emotions[emotion[0]] = [emotion[1]]
    eout = {}
    for e in sorted(emotions):
        eout[e] = sorted(emotions[e])
    return pout, eout

def read_texts(filename):
    '''
    
    returns list: [doc1, doc2, ...]
    '''
    txt = open(filename)
    out = []
    for line in txt:
        if line == '':
            continue
        out.append(line)
    return out

def loadVectors(filename:str):
    """This function loads word vectors from the file.

    :param filename:  the filename of the vectors
        (e.g. glove.subset.50d.txt)
    :return: a dictionary, key: token, value: list of numbers loaded
        from the file.
    """
    wordVecs = {}
    docs = open(filename, encoding="utf8")
    for line in docs:
        if line == '':
            continue
        wordlist = line.strip().split()
        vec = []
        broken = False
        for num in wordlist[1:]:
            try:
                float(num)
            except:
                broken = True
                break
            vec.append(float(num))
        if broken:
            continue
        wordVecs[wordlist[0].lower()] = vec
    vout = {}
    for w in sorted(wordVecs):
        vout[w] = wordVecs[w]
    return vout

def generate_questions(filename : str):
    '''
    

    returns emotion_questions, person_questions:
        {sentence1_missing_emotion: [emotion_word1, ...], ...}
        {sentence1_missing_person: [person_word1, person_word2, ...], ...}
    '''
    csvFile = pd.read_csv(filename)
    emotion_questions = {}
    person_questions = {}
    for i in range(len(csvFile)):
        em = csvFile["Emotion word"][i]
        if type(em)==str:
            q = re.sub(em.lower(), '________', csvFile["Sentence"][i].lower())
            if q in emotion_questions:
                emotion_questions[q].append(em)
            else:
                emotion_questions[q] = [em]
        emotion_questions[q] = list(set(emotion_questions[q]))
    for i in range(len(csvFile)):
        per = csvFile["Person"][i]
        q = re.sub(per.lower(), '________', csvFile["Sentence"][i].lower())
        q = re.sub('t________', 'the', q)
        q = re.sub('________artbreaking', 'heartbreaking', q)
        if q in person_questions:
            person_questions[q].append(per)
        else:
            person_questions[q] = [per]
        person_questions[q] = list(set(person_questions[q]))
    eqout = {}
    for q in sorted(emotion_questions):
        eqout[q] = sorted(emotion_questions[q])
    pqout = {}
    for q in sorted(person_questions):
        pqout[q] = sorted(person_questions[q])
    return eqout, pqout

class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        self.vectorizer = TfidfVectorizer(ngram_range=(1,3))
        self.matrix = self.vectorizer.fit_transform(texts)
        self.features = self.vectorizer.get_feature_names_out()

    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        labels = preprocessing.LabelEncoder()
        labels.fit(self.features)
        return int(labels.transform([feature]))

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        new_texts = self.vectorizer.transform(texts)
        return new_texts.toarray()

class VecDense:

    def tokenizeDoc(self, oneDoc: str):
        """This method tokenizes a a string.

        :param oneDoc: a string.
        :return: a tokenized sting.
        """
        sanitizedStr = re.sub(r'[^a-zA-Z0-9 ]', '', oneDoc)
        tokens = sanitizedStr.lower().split(" ")
        return tokens

    def getVecLength(self, vecIn: list):
        """This method computes the length of a vector.

        :param vecIn: a list representing a vector, one element per dimension.
        :return: the length of the vector.
        """
        return math.sqrt(sum([elem**2 for elem in vecIn]))

    def normalizeVec(self, vecIn:list):
        """This method normalizes a vector to unit length.

        :param vecIn:  a list representing a vector, one element per dimension.
        :return: a list representing a vector, that has been normalized to unit length.
        """
        return [elem/self.getVecLength(vecIn) for elem in vecIn]

    def dotProductVec(self, vecInA:list, vecInB:list):
        """This method takes the dot product of two vectors.

        :param vecInA, vecInB: two lists representing vectors,
            one element per dimension.
        :return: the dot product.
        """
        return sum([a*b for (a,b) in zip(vecInA, vecInB)])

    def cosine(self, vecInA: list, vecInB: list):
        """This method obtains the cosine between two vectors
            (which is nominally the dot product of two vectors of unit length).

        :param vecInA, vecInB: two lists representing vectors, one element per dimension.
        :return: the cosine.
        """
        return self.dotProductVec(self.normalizeVec(vecInA), self.normalizeVec(vecInB))

    def computeCentroidVector(self, tokensIn:list, vecDict:dict):
        """This method calculates the centroid vector from a list of
            tokens. The centroid vector is the "average"
            vector of a list of tokens.
        #NOTE:  Special considerations:
            - all tokens should be converted to lower case.
            - if a vector isn't in the dictionary, it
                shouldn't be a part of the average.

        :param tokensIn: a list of tokens.
        :param vecDict: the vector library is a dictionary, 'vecDict',
            whose keys are tokens, and values are lists representing vectors.
        :return: the centroid vector, represented as a list.
        """
        vecs = [vecDict[elem.lower()] for elem in tokensIn if elem in vecDict]
        avgs = []
        for i in range(len(vecs[0])):
            elems = [lst[i] for lst in vecs]
            avgi = sum(elems)/len(elems)
            avgs.append(avgi)
        return avgs

class Bias_Classification:

    def __init__(self, questions, tables, model, qtype, nanswers, text_filename):
        self.emotion_questions, self.person_questions = questions
        self.people, self.emotions = tables
        self.model = model
        self.qtype = qtype
        self.train_texts = None
        self.text_filename = text_filename
        self.nanswers = nanswers

    def search_emotions(self, question):
        '''
        
        returns general emotion of specific emotion word named in question text
        '''
        for key, value in self.emotions.items():
            for em in value:
                if em.lower() in question.lower():
                    return key
        return 'NA'

    def search_people(self, question):
        '''
        

        returns gender (if i=0) or race (if i=1) of person named in question text
        '''
        for key, value in self.people.items():
            if key.lower() == 'he' or key.lower() == 'she' or key.lower() == 'him' or key.lower() == 'her':
                continue
            if key.lower() in question.lower():
                return value
        if 'her' in question.lower() or 'she' in question.lower():
            return 'female', 'NA'
        if 'him' in question.lower() or 'he' in question.lower():
            return 'male', 'NA'

    def train(self):
        '''
        
        '''
        filename = self.text_filename
        if self.model == 'ngram':
            corpus = read_texts(filename)
        elif self.model == 'vector':
            corpus = loadVectors(filename)
        filename = filename.strip('testtrain_datatxt')
        filename = filename.strip('./')
        self.train_texts = corpus
        self.filename = filename

    def ngram_answer(self):
        '''
        
        returns dict: {question1: [first_choice, second_choice, ...], ...}
        '''
        if self.qtype == 'emotion':
            qdict = self.emotion_questions
        elif self.qtype == 'person':
            qdict = self.person_questions
        answers = {}
        vocab = TextToFeatures(self.train_texts)
        self.length = len(vocab.features)
        for question in sorted(qdict):
            choices = qdict[question]
            new_texts = [re.sub('________', c, question) for c in choices]
            features = vocab(new_texts)
            sums = np.sum(features, 1)
            ind = (-sums).argsort()[:self.nanswers]
            answers[question] = [choices[i] for i in ind]
        self.answers = answers

    def vector_answer(self):
        '''
        

        returns dict: {question1: [first_choice, second_choice, ...], ...}
        '''
        vecDense = VecDense()
        
        if self.qtype == 'emotion':
            qdict = self.emotion_questions
        elif self.qtype == 'person':
            qdict = self.person_questions

        answers = {}
        for q in sorted(qdict):
            c = qdict[q]
            qtokens = vecDense.tokenizeDoc(q)
            qvec = vecDense.computeCentroidVector(qtokens, self.train_texts)
            cosines = []
            new_texts = [re.sub('________', choice, q) for choice in c]
            for choice in new_texts:
                ctokens = vecDense.tokenizeDoc(choice)
                cvec = vecDense.computeCentroidVector(ctokens, self.train_texts)
                cosines.append(vecDense.cosine(qvec, cvec))
                npcos = np.asarray(cosines)
            ind = (-npcos).argsort()[:self.nanswers]
            answers[q] = [c[i] for i in ind]
        self.length = len(self.train_texts)
        self.answers = answers

    def score_answers(self):
        '''


        returns dict:
            {emotion1: {person_category1: count, ...}, ...} if qtype is emotion
            {person_category1: {emotion1: count, ...}, ...} if qtype is person
        '''
        if self.qtype == 'emotion':
            gcounts = {}
            rcounts = {}
            for q,a in self.answers.items():
                weight = len(a)
                for em in a:
                    emotion = self.search_emotions(em)
                    cat = self.search_people(q)
                    if emotion not in gcounts:
                        gcounts[emotion] = {'male': 0, 'female': 0}
                    if emotion not in rcounts:
                        rcounts[emotion] = {'African-American': 0, 'European': 0, 'NA': 0}
                    gcounts[emotion][cat[0]] += weight
                    rcounts[emotion][cat[1]] += weight
                    weight -= 1
        elif self.qtype == 'person':
            gcounts = {}
            rcounts = {}
            for q,a in self.answers.items():
                weight = len(a)
                for p in a:
                    cat = self.people[p][0]
                    emotion = self.search_emotions(q)
                    if cat not in gcounts:
                        gcounts[cat] = {'joy': 0, 'fear': 0, 'anger': 0, 'sadness': 0, 'NA': 0}
                    gcounts[cat][emotion] += weight
                    weight -= 1
                weight = len(a)
                for p in a:
                    cat = self.people[p][1]
                    emotion = self.search_emotions(q)
                    if cat not in rcounts:
                        rcounts[cat] = {'joy': 0, 'fear': 0, 'anger': 0, 'sadness': 0, 'NA': 0}
                    rcounts[cat][emotion] += weight
                    weight -= 1
        self.gcounts = gcounts
        self.rcounts = rcounts

    def score_model(self):
        '''
        
        '''
        rchisq = 0
        gchisq = 0
        if self.qtype == 'emotion':
            for em, val in self.gcounts.items():
                scores = list(val.values())
                if len(scores) > 2:
                    scores = scores[0:2]
                mean = st.mean(scores)
                for item in scores:
                    gchisq += (item-mean)**2/mean
            for em, val in self.rcounts.items():
                scores = list(val.values())
                if len(scores) > 2:
                    scores = scores[0:2]
                mean = st.mean(scores)
                for item in scores:
                    rchisq += (item-mean)**2/mean
        elif self.qtype == 'person':
            for per, val in self.gcounts.items():
                scores = list(val.values())
                if len(scores) > 4:
                    scores = scores[0:4]
                mean = st.mean(scores)
                for item in scores:
                    gchisq += (item-mean)**2/mean
            for per, val in self.rcounts.items():
                scores = list(val.values())
                if len(scores) > 4:
                    scores = scores[0:4]
                mean = st.mean(scores)
                for item in scores:
                    rchisq += (item-mean)**2/mean
        return rchisq,gchisq

    def answer_questions(self):
        if self.model == 'ngram':
            self.ngram_answer()
        elif self.model == 'vector':
            self.vector_answer()

    def output_rows(self):
        '''
        

        returns list: [['model', 'qtype', 'filename', 'category', 'value', 'emotion', 'count_score', 'model_score'], ...]
        '''
        table = []

        self.answer_questions()
        self.score_answers()
        
        rchisq,gchisq = self.score_model()
        if self.qtype == 'emotion':
            for em, dict in self.gcounts.items():
                for g, count in dict.items():
                    table.append([self.model, self.qtype, self.filename, 'gender', g, em, count, gchisq, self.length])
            for em, dict in self.rcounts.items():
                for r, count in dict.items():
                    table.append([self.model, self.qtype, self.filename, 'race', r, em, count, rchisq, self.length])
        if self.qtype == 'person':
            for per, dict in self.gcounts.items():
                for em, count in dict.items():
                    table.append([self.model, self.qtype, self.filename, 'gender', per, em, count, gchisq, self.length])
            for per, dict in self.rcounts.items():
                for em, count in dict.items():
                    table.append([self.model, self.qtype, self.filename, 'race', per, em, count, rchisq, self.length])
        return table
    
    def write_csv(self, out_filename):
        
        self.train()
        rows = self.output_rows()

        with open(out_filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
        csvfile.close()

        qfile = "answers/" + self.filename + "_" + self.qtype + ".csv"
        title = ["question"]
        for i in range(self.nanswers):
            title.append("answer" + str(i + 1))
        qout = []
        for q, c in self.answers.items():
            qrow = [q] + c
            qout.append(qrow)
        
        with open(qfile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(title)
            csvwriter.writerows(qout)
        csvfile.close()
    

def main():
    
    outfile = 'bias_results.csv'
    
    '''
    ##################### ONLY RUN IF REPLACING DATA ###########################
    if os.path.exists(outfile):
        os.remove(outfile)

    fields = ['model', 'qtype', 'filename', 'category', 'value', 'emotion', 'count_score', 'model_score', 'corpus_length']
    with open(outfile, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
    csvfile.close()
    ############################################################################
    '''

    questions = generate_questions("data/Equity-Evaluation-Corpus.csv")
    tables = lookup_tables("data/Equity-Evaluation-Corpus.csv")

    print('Loaded questions')

    filenames = ['train_data/' + str(file) for file in os.listdir('train_data')]

    for i in range(len(filenames)):
        if 'text' in filenames[i]:
            model = 'ngram'
        elif 'glove' in filenames[i]:
            model = 'vector'

        class_obj = Bias_Classification(questions, tables, model, 'emotion', 3, filenames[i])
        class_obj.write_csv(outfile)

        print((class_obj.filename + ' loaded').ljust(30) + str(round((i+1)/len(filenames)*100, 2)) + '% complete')
    
main()
