from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd
from utilities import *

def doit(self, text: str) -> List[List[str]]:
    text=text.lower()
    text=re.sub(r'[^.\w\s]', '', text)
    list=text.split('.')
    result=[]
    for i in list:
        result.append(self.tokenize(i))
    return result




class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK(0.1)
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])

    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)

    def correct(self, text1: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        ## I am assuming that i am having only one sentence as a input
        ## It is checking if all the words are present in the dictationary

        for x in range (len(text1)):
            text=text1[x].split()
            
            l=[]
            index=-1
            for i in range(len(text)):
                if(text[i] not in self.internal_ngram.dictationary):
                    index=i
                    l.append(index)

            

            ## It is checking that the (if all the words are present) which one of them is most probably wrong
            context=""
            count=0
            if(index==-1):
                mini=10000
                for i in range(len(text)-1):
                    temp_score=self.internal_ngram.get_prob(context,text[i],self.internal_ngram.n)
                    temp_context=context+" "+text[i]
                    temp_score+=self.internal_ngram.get_prob(temp_context,text[i+1],self.internal_ngram.n)
                    if(mini>temp_score):
                        mini=temp_score
                        index=i
                    if context!="":
                        context+=" "
                    context+=text[i]
                    count+=1
                    if(count==self.internal_ngram.n):
                        count-=1
                        context=context[len(text[(i+1)-self.internal_ngram.n])+1:]
                l.append(index)

            for i in range(len(l)):
                self.correct_index(l[i],text1,text,x)
            

            ## there will be an assertion to check if the output text is of the same
            ## length as the input text
        return text1
    
    def correct_index(self,index,text1,text,x):
        ## Now, we have to correctly identify all the words with edit distance<2 and give them probabilities
            result=[]
            probs=[]
            for i in self.internal_ngram.dictationary:
                if(edit_distance(i,text[index])==2):
                    result.append(i)
                    probs.append(0.1)
                if(edit_distance(i,text[index])==1):
                    result.append(i)
                    probs.append(0.9)

            ## i have to figure out the contexts, we would be checking the frequencies





            context=""
            count=0
            next_word=""
            for i in range(max(0,index-self.internal_ngram.n+1),index):
                context+=text[i]
                count+=1
                if(i!=index-1):
                    context+=' '
                if(i+3<len(text)):
                    next_word=text[i+2]
            

            

            ## Now we have to multiply all the probabilities to get the final one
            mx=0
            final_index=0
            for i in range(len(result)):
                temp_context=''
                if(reduce(context)==''):
                    temp_context=result[i]
                else:
                    temp_context=reduce(context)+" "+result[i]
                score=self.internal_ngram.get_prob(context,result[i],self.internal_ngram.n)+self.internal_ngram.get_prob(temp_context,next_word,self.internal_ngram.n)
                probs[i]=probs[i]*score
                if(probs[i]>mx):
                    mx=probs[i]
                    final_index=i
            if(len(result)!=0):
                text[index]=result[final_index]
            
            s=""
            for i in range(len(text)):
                s=s+text[i]
                if(i!=len(text)-1):
                    s=s+" "
            text1[x]=s
    

# if __name__=="__main__":
#     spl=SpellingCorrector()
    
#     # spl.internal_ngram.update()

#     with open("data/train1.txt", "r") as file:
#         test_sentence = file.read()
#     with open("data/train2.txt", "r") as file:
#         test_sentence += file.read()

#     spl.internal_ngram.fit(spl.internal_ngram.doit(test_sentence))

#     error_text=["he is going hime to tll his wime and kids"]
#     print(spl.correct(error_text))