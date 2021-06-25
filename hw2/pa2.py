import pandas as pd
import numpy as np
import os
import re
import math
from os import listdir
from os.path import isfile, isdir, join
import sys
import nltk
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

def toTerms(fileName):
    # read file
    fileContent = open(fileName).read()
    
    # Tokenization.
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    digital = '0123456789'
    for p in punctuation:
        fileContent = fileContent.replace(p,' ')
    for d in digital:
        fileContent = fileContent.replace(d,' ')
    tokens = str.split(fileContent)
    
    # Lowercasing everything.
    lowercase = [lowercase.lower() for lowercase in tokens]
    
    # Stemming using Porter's algorithm
    # https://github.com/jedijulia/porter-stemmer/blob/master/stemmer.py
    porterStemmer = PorterStemmer() 
    stemming = [porterStemmer.stem(word) for word in lowercase]
    
    # Stopword removal.
    # stopWords = set(stopwords.words('english')) 
    # stopWords = str.split(open("stopwords_en.txt").read())
    stopWords = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']
    answer = []
    for word in stemming:
        if word not in stopWords:
            answer.append(word)
    return answer

# Q1
def createDictionary():
    print("creating Dictionary...")
    mypath = "./IRTM"
    files = listdir(mypath)

    counter = 0
    for f in files:
        fullpath = join(mypath, f)
        counter += 1
        if fullpath.find(".txt") == -1: 
            continue 

        if counter == 1: 
            docTermsList = toTerms(fullpath)
            docTerms = pd.DataFrame (docTermsList,columns=['term'])
            docTerms = docTerms.groupby('term').size().reset_index(name = 'tf')
            docTerms.insert(1,'df',0)
            continue

        tempTermsList = toTerms(fullpath)
        tempTerms = pd.DataFrame(tempTermsList,columns=['term'])
        tempTerms = tempTerms.groupby('term').size().reset_index(name = 'tf')


        docTerms = docTerms.merge(tempTerms, how= "outer", on ="term") 
        docTerms['df'].fillna(value = 0, inplace =True)
        # count: the number of none NaN columns, so must minus 2 (term, df)  
        docTerms['df'] += docTerms.count(axis = "columns").values - 2  
        docTerms.drop(docTerms.columns[2:4],axis =1,inplace = True) 
        
    docTerms = docTerms.sort_values(by=['term'])
    docTerms['df'] = docTerms['df'].astype(int)
    docTerms.insert(0,'t_index',0)
    docTerms['t_index'] = range(1, docTerms.shape[0]+1)
    docTerms.to_csv(r'dictionary.txt', header=True, index=False, sep=' ', mode='w')
    print("Finish Create Dictionary.")
    return docTerms

# Q2
def toUnitVector(originalFileName, vectorFileName):
    mypath = "./IRTM/"
    fullpath = mypath + originalFileName
    termsList = toTerms(fullpath)
    
    tempTerms= pd.DataFrame (termsList,columns=['term'])
    tempTerms = tempTerms.groupby('term').size().reset_index(name = 'tf')
    tempTerms.insert(1,'tf-idf',0) 

    tempTerms = tempTerms.merge(docTerms, how= "left", on ="term")
    tempTerms['tf-idf'] = tempTerms['tf'] * np.log10(1095/tempTerms['df'])
    tempTerms.drop(columns=['term', 'tf', 'df'],inplace = True) 
    tempTerms= tempTerms[[ 't_index' , 'tf-idf']]
    tempTerms = tempTerms.sort_values(by=['t_index'])
    tempTerms['tf-idf'] = tempTerms['tf-idf'] / (tempTerms['tf-idf']**2).sum()**0.5
    
    f = open(vectorFileName, "w")
    f.write(str(tempTerms.shape[0])+"\n")
    f.close()
    tempTerms.to_csv(vectorFileName, header=True, index=False, sep=' ', mode='a')

# Q3
def readUnitVectorTxt(originalFileName):
    vectorFileName = "doc" + originalFileName
    toUnitVector(originalFileName, vectorFileName)
    f = open(vectorFileName)
    line = f.readline()
    counter = 0
    t_indexs = []
    tfIdf = []
    while line:
        if counter == 0 or counter == 1:
            line = f.readline()
            counter += 1
            continue
        token = str.split(line)
        t_indexs.append(int(token[0]))
        tfIdf.append(float(token[1]))
        line = f.readline()
        counter += 1
    f.close()
    vector = { "t_indexs" : t_indexs,
                "tf-idf" : tfIdf }
    Vector = pd.DataFrame(vector)
    return Vector

def cosine(originalFileName_X, originalFileName_Y):
    unitVectorX = readUnitVectorTxt(originalFileName_X)
    unitVectorY = readUnitVectorTxt(originalFileName_Y)
    
    Vector = unitVectorX.merge(unitVectorY, how= "inner", on ="t_indexs")
    cosineSimilarity = (Vector['tf-idf_x'] *  Vector['tf-idf_y']).sum()
    return cosineSimilarity

if __name__ == "__main__":

	filename_X = sys.argv[1]
	filename_Y = sys.argv[2]
	print("originalFileName:" + filename_X + " and " + filename_Y)
	docTerms = createDictionary()
	cosineSimilarity = cosine(filename_X, filename_Y)

	print("cosine Similarity: ", cosineSimilarity)
