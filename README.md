# [Bias in Pre-Trained Language Models](https://github.com/sbeecher0575/nlpbias/blob/main/paper.pdf)

Here is the code and results for a more extended look at this research.

## answers

This folder contains the .csv files of questions and their ordered 3 answers (first is most likely answer) for each model and question type (person vs. emotion). I only used the emotion questions in the paper.

## data

This folder contains the sentence text and name tagging information from which I created my question list. It is from Kiritchenko and Mohammad, 2018 [Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems](https://aclanthology.org/S18-2005/), cited in my paper.

## train_data

This folder would be the training texts I used to train the models, but they are too large to fit here (even zipped). Instead I link to the locations from which I downloaded them myself.

## graphs

This folder contains graphs and plots used in the paper and extra ones that did not fit, but I felt were beneficial. It also contains the R file I used to create them.

## paper.pdf

The paper describing the project, theory, sources, and mathematical derivations.

## bias_question.py

This python program creates the questions from the sentence text, trains the models on the training data, and writes output files to the answers folder and final counts to bias_results.csv

## bias_results.csv

This .csv file contains the count scores of all emotions-race/gender combinations for every model and question type. This file also contains the chi-square statistics and corpus length for each model (either number of vectors or number of unique tokens).
