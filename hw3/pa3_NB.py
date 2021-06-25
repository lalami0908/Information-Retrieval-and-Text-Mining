import nltk
from nltk.corpus import stopwords
import string
import math

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
    porterStemmer = nltk.PorterStemmer() 
    stemming = [porterStemmer.stem(word) for word in lowercase]
    
    # Stopword removal.
    stopWords = set(stopwords.words('english')) 
    # stopWords = str.split(open("stopwords_en.txt").read())
    answer = []
    for word in stemming:
        if word not in stopWords:
            answer.append(word)
    return answer
# ===========================================================================

classes = [ ["11","19","29","113","115","169","278","301","316","317","321","324","325","338","341"],
			["1","2","3","4","5","6","7","8","9","10","12","13","14","15","16"],
			["813","817","818","819","820","821","822","824","825","826","828","829","830","832","833"],
			["635","680","683","702","704","705","706","708","709","719","720","722","723","724","726"],
			["646","751","781","794","798","799","801","812","815","823","831","839","840","841","842"],
			["995","998","999","1003","1005","1006","1007","1009","1011","1012","1013","1014","1015","1016","1019"],
			["700","730","731","732","733","735","740","744","752","754","755","756","757","759","760"],
			["262","296","304","308","337","397","401","443","445","450","466","480","513","533","534"],
			["130","131","132","133","134","135","136","137","138","139","140","141","142","143","145"],
			["31","44","70","83","86","92","100","102","305","309","315","320","326","327","328"],
			["240","241","243","244","245","248","250","254","255","256","258","260","275","279","295"],
			["535","542","571","573","574","575","576","578","581","582","583","584","585","586","588"],
			["485","520","523","526","527","529","530","531","532","536","537","538","539","540","541"]]

classToken=list() #list of list of list (class-docs-tokens) 13 * 15* token
terms=dict() #store all the terms and the LLR for classes 5212
trainDocs=list() #train docs

path="IRTM/"
extension=".txt"
for lists in classes:
	token_list=list()
	for items in lists:
		trainDocs.append(int(items))
		fileName=path+str(items)+extension
		tokens=toTerms(fileName)
		for tok in tokens:
			if(tok not in terms.keys()): 
				terms[tok]=list()
		token_list.append(tokens) 
	classToken.append(token_list)

classCnt=13
#================  log likelihood ratio feature selection ====================
for t in terms.keys():
	for i in range(classCnt): #run through each class #13
		n11=0 
		n10=0 
		n01=0 
		n00=0 

		#check on topic
		for doc_token in classToken[i]:
			if(t in doc_token): #present
				n11+=1
			else:
				n10+=1

		#check off topic		
		for j in range(classCnt):
			if(j!=i):
				for doc_token in classToken[j]:
					if(t in doc_token): #present
						n01+=1
					else:
						n00+=1
		N=n11+n10+n01+n00		
		pt= (n11+n01)/N #pt for Hypothesis 1
		p1= n11/(n11+n10) #p1 for Hypothesis 2
		p2= n01/(n01+n00) #p2 for Hypothesis 2
		LR=((pt**n11)*((1-pt)**n10)*(pt**n01)*((1-pt)**n00))/((p1**n11)*((1-p1)**n10)*(p2**n01)*((1-p2)**n00))
		LLR=(-2)*math.log2(LR)
		terms[t].append(LLR)
# ===============  get 500 higher average LLR as tokens, ignore not filtered, create newClassToken =================================

VocSize=500

dictLLR=dict() 
filterTerms=dict()
newClassToken=list() #new tokens for docs , list of list of list (class-docs-tokens) 13* 15 *500

#average the LLRs and get the higher 500
for t in terms.keys():
	dictLLR[t]=sum(terms[t])/len(terms[t])

for key in sorted(dictLLR,key=dictLLR.get,reverse=True)[:VocSize]:
		filterTerms[key]=list()

#save new token, ignoring tokens not filtered
for eachClass in classToken:
	newClass=list()
	for eachDoc in eachClass:
		newDoc=list()
		for eachTerm in eachDoc:
			if(eachTerm in filterTerms.keys()):
				newDoc.append(eachTerm)
		newClass.append(newDoc)
	newClassToken.append(newClass)

# ================ Training =====================
for term in filterTerms.keys():
	for eachClass in newClassToken:
		tokenCntInClass=0
		appearance=0
		for eachDoc in eachClass:
			tokenCntInClass+=len(eachDoc) #record total tokens in class
			for eachTerm in eachDoc:
				if(term==eachTerm):
					appearance+=1 #occurence of the term in a class
		prob=(appearance+1)/(VocSize+tokenCntInClass) # add-one smoothing.
		filterTerms[term].append(prob) #record the prob in filterTerms
# =================================================

# ================ Testing =======================
result={}
for i in range(1,1096):
	if i not in trainDocs: 
		predicts=[math.log2(1/13)]*13
		fileName=path+str(i)+extension
		tokens=toTerms(fileName) 
		for word in tokens:
			if(word in filterTerms.keys()): 
				for c in range(classCnt):
					predicts[c]+=math.log2(filterTerms[word][c]) 

		maxValue=max(predicts) 
		result.setdefault(str(i),str(predicts.index(maxValue)+1))
#=================================================

#================== Write =========================
with open('result.csv', 'w') as f:
	f.write("%s,%s\n"%('Id','Value'))
	for key in result.keys():
		f.write("%s,%s\n"%(key,result[key]))
f.close()

#=================================================