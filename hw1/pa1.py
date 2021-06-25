import sys
import nltk
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

if __name__ == "__main__":
	# read file
	filename = sys.argv[1]
	file_content = open(filename).read()

	# Tokenization.
	file_content = file_content.replace('.',' ')
	punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
	for p in punctuation:
	    file_content = file_content.replace(p,'')
	tokens = str.split(file_content)

	# Lowercasing everything.
	lowercase = [lowercase.lower() for lowercase in tokens]

	# Stemming using Porter's algorithm
	# https://github.com/jedijulia/porter-stemmer/blob/master/stemmer.py
	porterStemmer = PorterStemmer() 
	stemming = [porterStemmer.stem(word) for word in lowercase]

	# Stopword removal.
	stopWords = set(stopwords.words('english')) 
	answer = []
	for word in stemming:
	    if word not in stopWords:
	        answer.append(word)

	# Save the result as a txt file. 
	for i in range(len(answer)):
	    write = open('result.txt','a') 
	    write.write(answer[i]) 
	    if i != len(answer)-1:
	        write.write("\n") 
	    write.close()