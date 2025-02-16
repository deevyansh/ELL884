import numpy as np
import pandas as pd
from typing import List
import re
import math


# config.py

class NGramBase:
    def __init__(self):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.n=3
        self.current_config = {}
        self.count_con=[{} for i in range (self.n+1)]
        self.count_word=[{} for i in range (self.n+1)]
        self.all_words={}
        self.word_count=0
        self.dictationary={}

    def method_name(self) -> str:

        return f"Method Name: {self.current_config['method_name']}"

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        for i in data:
            for j in i:
                self.word_count+=1
                if(j in self.dictationary):
                    self.dictationary[j]+=1
                else:
                    self.dictationary[j]=1


        for k in range(2,self.n+1):
            for i in data:
                curr_context = "" # Use a list
                count=0
                for j in range (0,len(i)):
                    self.all_words[i[j]]=1
                    if curr_context not in self.count_con[k]:
                        self.count_con[k][curr_context]=1
                    else:
                        self.count_con[k][curr_context]+=1
                    if (curr_context,i[j]) not in self.count_word[k]:
                        self.count_word[k][(curr_context,i[j])]=1
                    else:
                        self.count_word[k][(curr_context,i[j])]+=1
                    if count==((k)-1):
                        curr_context=curr_context[len(i[j-(k-1)])+1:]
                        count-=1
                    count+=1
                    if(curr_context!=''):
                        curr_context+=' '+i[j]
                    else:
                        curr_context=i[j]

        if hasattr(self, 'update'):
            self.update()


    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        return text.split()

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
        raise NotImplementedError

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()
    
    def get_prob(self,curr_context,word,level) ->float:
        if(level==1):
            return self.dictationary[word]/self.word_count
        return self.count_word[level][(curr_context,word)]/self.count_con[level][curr_context]


    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        result=self.doit(text)
        sum=0
        word_count=0
        for i in result:
            curr_context=""
            count=0
            for j in range (len(i)):
                # print(self.count_word[self.n][(curr_context,i[j])]/self.count_con[self.n][curr_context])
                prob=math.log10(self.get_prob(curr_context,i[j],self.n))
                sum+=prob
                word_count+=1
                if count==((self.n)-1):
                    curr_context=curr_context[len(i[j-(self.n-1)])+1:]
                    count-=1
                count+=1
                if(curr_context!=''):
                    curr_context+=' '+i[j]
                else:
                    curr_context=i[j]

        if(word_count==0):
            return 0
        sum/=word_count
        sum=-sum
        sum= 10**(sum)
        return sum
    
    
    def doit(self, text: str) -> List[List[str]]:
        text = text.lower()
        text = re.sub(r'[^.\w\s]', '', text)  # Remove punctuation except periods
        sentences = text.split('.')  # Split text into sentences
        result = []

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            result.append(tokens)

        return result

# if __name__ == "__main__":
    # tester_ngram = NGramBase()
    # with open("data/train1.txt", "r") as file:
    #     test_sentence = file.read()
    # with open("data/train2.txt", "r") as file:
    #     test_sentence += file.read()
    # tester_ngram.fit(tester_ngram.doit(test_sentence))

    # # for i in tester_ngram.count_con:
    # #     print(i)
    # # for i in tester_ngram.count_word:
    # #     print(i)

    
    # print(tester_ngram.perplexity("It urged that the next Legislature"))

    # test_sentence="This is a test sentence"
    # tester_ngram.fit(tester_ngram.doit(test_sentence))
    # print(tester_ngram.count_word)
    # print(tester_ngram.get_prob("a","test",2))

