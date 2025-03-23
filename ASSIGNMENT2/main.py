### Importing the libraries ####
import torch
import torch.optim as optim
from config import *
from bilstm_crf import BiLSTM_CRF

import pandas as pd
import numpy as np
import ast
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec


### Importing the data ####
df = pd.read_csv("data/ner_train.csv").head(1000).reset_index(drop=True)

print(df.head())
print(df.iloc[0])

df['Sentence']=df['Sentence'].apply(lambda x: x.split())
df['POS'] = df['POS'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Tag'] = df['Tag'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


#### WORD 2 VEC MODEL TRAINING ####
word_model=Word2Vec(df['Sentence'],vector_size=e1_size-1,window=5,min_count=1,workers=4)
word_model.save("word2vec_model.model")


#### Remove the wrong indicies from the the df ####
wrong_indices=[]
for i in range(len(df)):
    if(len(df.iloc[i]['Sentence']) !=len(df.iloc[i]['POS'])):
        wrong_indices.append(i)

df = df.drop(wrong_indices).reset_index(drop=True)


#### Transforming the sentences into disred forms ####
sentences=[]
for i in range(len(df)):
    l=[]
    for j in range (len(df.iloc[i]['Sentence'])):
        l.append([df.iloc[i]['Sentence'][j], df.iloc[i]['POS'][j], df.iloc[i]['Tag'][j]])
    sentences.append(l)


#### Making the length of all sentences same ####
max_sent_size=max([len(sent) for sent in sentences])
mask=[[1]*len(sent)+[0]*(max_sent_size-len(sent)) for sent in sentences]
sentences = [sent + [["pad", "pad", "pad"]] * (max_sent_size - len(sent)) for sent in sentences]
print("A particular set of sentence",sentences[0])



#### Function used to make the sentences in the numerical form####
pos_to_ix={}
for sent in sentences:
    for token,postag,y in sent:
        if(postag not in pos_to_ix):
            pos_to_ix[postag]=len(pos_to_ix)


def word2features(sent, i):
    word_embedding = word_model.wv[sent[i][0]] if sent[i][0] in word_model.wv else np.zeros(e1_size-1)
    pos_index = pos_to_ix[sent[i][1]]  if sent[i][1] in pos_to_ix else -1

    return np.append(word_embedding, pos_index)

def sent2features(sent):
    return np.array([word2features(sent, i) for i in range(len(sent))])



### Making the sentence labels in the numerical forms####

tag_to_ix={"pad":pad_tag_id,"start":start_tag_id, "end":end_tag_id} ## Pad and unk has been removed for now
ix_to_tag={pad_tag_id:"pad",start_tag_id:"start",end_tag_id:"end"} ## pad and unk has been removed for now
for sent in sentences:
    for token,postag,y in sent:
        if(y not in tag_to_ix):
            ix_to_tag[len(tag_to_ix)]=y
            tag_to_ix[y]=len(tag_to_ix)
            
print(tag_to_ix)

def sent2labels(sent):
    return [tag_to_ix[y]  for [token, postag, y] in sent]

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]



#### Trianing the model using the PyTorch  ####

class NERDataset(Dataset):
    def __init__(self,X,y,mask):
        self.X=torch.tensor(X, dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
        self.mask=torch.tensor(mask,dtype=torch.int)

    def __len__(self):
        return len(X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]
    

dataset=NERDataset(X,y,mask)
batch_size=50
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model=BiLSTM_CRF(e1_size,len(tag_to_ix))
optimizer=optim.SGD(model.parameters(),lr=0.001,weight_decay=1e-2)


for epoch in range(300):
    total_loss=0

    for batch in dataloader:

        model.zero_grad()
        X_tensor,y_tensor,mask_tensor=batch
        loss=model.loss(X_tensor,y_tensor,mask_tensor)
        total_loss+=loss
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch,"Loss:",total_loss)



## Printing the final results over the training data after training ####

def ids_to_tags(seq, itos):
    l=[]
    for x in seq:
        x_int=x.item()
        l.append(itos[x_int])
    return l

print("Prediction after training")
with torch.no_grad():
    X_tensor=torch.tensor(X,dtype=torch.float32)
    mask_tensor=torch.tensor(mask,dtype=torch.int)
    scores, seqs = model(X_tensor, mask=mask_tensor)
    y_flat = [label for seq in y for label in seq]
    seqs_flat = [label for seq in seqs for label in seq]
    print(classification_report(y_flat,seqs_flat))
    for score, seq in zip(scores, seqs):
        str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
        # print('%.2f: %s' % (score.item(), str_seq))


#### Calculating results on the testing data ####

df_test=pd.read_csv("data/ner_test.csv")
df_test['Sentence']=df_test['Sentence'].apply(lambda x: x.split())
df_test['POS'] = df_test['POS'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_test['Tag'] = df_test['Tag'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


sentences=[]
for i in range(len(df_test)):
    l=[]
    for j in range (len(df_test.iloc[i]['Sentence'])):
        l.append([df_test.iloc[i]['Sentence'][j], df_test.iloc[i]['POS'][j], df_test.iloc[i]['Tag'][j]])
    sentences.append(l)

max_sent_size=max([len(sent) for sent in sentences])
mask=[[1]*len(sent)+[0]*(max_sent_size-len(sent)) for sent in sentences]
sentences = [sent + [["pad", "pad", "pad"]] * (max_sent_size - len(sent)) for sent in sentences]
print("A particular set of sentence",sentences[0])


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

with torch.no_grad():
    X_tensor=torch.tensor(X,dtype=torch.float32)
    mask_tensor=torch.tensor(mask,dtype=torch.int)
    scores, seqs = model(X_tensor, mask=mask_tensor)
    y_flat = [label for seq in y for label in seq]
    seqs_flat = [label for seq in seqs for label in seq]
    print(classification_report(y_flat,seqs_flat))
    for score, seq in zip(scores, seqs):
        str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
        # print('%.2f: %s' % (score.item(), str_seq))

