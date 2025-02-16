from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd
import math

class NoSmoothing(NGramBase):

    def __init__(self):
        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

class AddK(NGramBase):
    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)
        
    def get_prob(self,context,word, level):
        vac_size=self.k*len(self.count_con[self.n])
        if context not in self.count_con[level]:
            return self.k/vac_size
        if ((context,word) in self.count_word[level]):
            p=((self.count_word[self.n][(context,word)]+self.k)*(self.count_con[self.n][context]))/(self.count_con[self.n][context]+vac_size)
            return p/self.count_con[self.n][context]
        else:
            p=(self.k*self.count_con[self.n][context]/(self.count_con[self.n][context]+vac_size))
            return p/self.count_con[self.n][context]
        
class StupidBackoff(NGramBase):
    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        self.alpha=0.4


    def reduce(self,context: str) ->str:
        index=len(context)
        for i in range(len(context)):
            if(context[i]==' '):
                index=i
                break
        return context[index+1:]

    def get_prob(self,curr_context, word, level):
        if(level==0):
            return 0.00001
        if(level==1):
            if(word in self.dictationary):
                return self.dictationary[word]/self.word_count
            else:
                return self.alpha*self.get_prob(self.reduce(curr_context),word,0)
        if((curr_context,word) in self.count_word[level]):
            return (self.count_word[level][(curr_context,word)]/self.count_con[level][curr_context])
        else:
            return self.alpha*self.get_prob(self.reduce(curr_context),word,level-1)
        

class GoodTuring(NGramBase):

    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
        self.Nmap={}

    def update(self):
        for i in range(0,10):
            self.Nmap[i]=0

        sum=0
        for i in self.count_word[self.n]:
            if(self.count_word[self.n][i] in self.Nmap):
                self.Nmap[self.count_word[self.n][i]]+=1
            else:
                self.Nmap[self.count_word[self.n][i]]=1
            sum+=self.count_word[self.n][i]
        self.Nmap[0]=(self.word_count**2)-sum


    def calci(self,x):
        if((x+1) not in self.Nmap):
            return x
        
        else:
            x=((x+1)*self.Nmap[x+1]/self.Nmap[x])
            return x
        
    def get_prob(self,curr_context,word,level):
        if((curr_context,word) not in self.count_word[self.n]):
            return self.calci(0)/self.count_con[self.n][curr_context]
        else:
            return self.calci(self.count_word[self.n][(curr_context,word)])/self.count_con[self.n][curr_context]


class Interpolation(NGramBase):

    def __init__(self,l):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)

    def reduce(self,context: str) ->str:
        index=len(context)
        for i in range(len(context)):
            if(context[i]==' '):
                index=i
                break
        return context[index+1:]
        
    def get_prob(self,context,word,level):
        prob=1e-10
        while(level!=0):
            if((context,word) not in self.count_word[level]):
                prob+=0
            else:
                prob+=(self.l[level]*self.count_word[level][(context,word)]/self.count_con[level][context])
            level-=1
            # print(context,level,prob)
            context=self.reduce(context)
        return prob


class KneserNey(NGramBase):
    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        self.Pmap={}
        self.Cmap={}

    def update(self):
        for i in self.count_word[self.n]:
            j=i[0]
            i=i[1]
            # print(j,"next",i)
            if(i not in self.Pmap):
                self.Pmap[i]=1
            else:
                self.Pmap[i]+=1
            if (j not in self.Cmap):
                self.Cmap[j]=1
            else:
                self.Cmap[j]+=1

    def get_prob(self,context,word,level):
        if(context not in self.count_con[self.n]):
            return 0
        # print(self.Cmap[context])
        lam=(self.d*self.Cmap[context])/(self.count_con[level][context])
        # print(self.Pmap[word])
        if(word in self.Pmap):
            Pconti=(self.Pmap[word]/len(self.count_word[self.n]))*lam
        else:
            Pconti=0
        if((context,word) not in self.count_word[self.n]):
            return Pconti
        score=max(self.count_word[level][(context,word)]-self.d,0)/self.count_con[level][context]
        score+=Pconti
        return score



# if __name__=="__main__":
#     tester_ngram = Interpolation()
#     test_sentence="This is a test sentence. This is a second pronoun"
#     tester_ngram.fit(tester_ngram.doit(test_sentence))

#     # with open("data/train1.txt", "r") as file:
#     #     test_sentence = file.read()
#     # with open("data/train2.txt", "r") as file:
#     #     test_sentence += file.read()
#     # tester_ngram.fit(tester_ngram.doit(test_sentence))


#     # tester_ngram.update()
#     print(tester_ngram.count_word)
#     print(tester_ngram.get_prob("a","test",2))

