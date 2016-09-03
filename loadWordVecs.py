import numpy as np
import cPickle 
import time
from optparse import OptionParser
from nltk.tokenize import wordpunct_tokenize

#parser = OptionParser()
#parser.add_option("-o", "--output", dest="outputFile",
#                  help="Weights output file name",default = '../WordVecFiles/wordToIndex.dict')
#parser.add_option("-p", "--pretrained", dest="pretrained",
#                  help="Pretrained word vector file name",default = '../WordVecFiles/glove.6B.300d.txt')
#
#(options, args) = parser.parse_args()

def extractVecs(pretrained):
## Pandas read_csv breaks while reading text file. Very buggy. Manually read each line.
	t0 = time.clock()
	with open(pretrained,'r') as f:
	        content = [item.rstrip().lower().split(' ') for item in f.readlines()]

	globalWordFile = np.asmatrix(content,dtype = str)
	globalWordTokens = globalWordFile[:,0].astype('str')
	globalWordVectors = globalWordFile[:,1:].astype(np.float)
	globalWordFile = None
	
	print time.clock() - t0, " seconds taken for loading and slicing gLoVe Word Vectors"
	return globalWordTokens,globalWordVectors

def generateWordDict(word_tokens,pretrained = '../WordVecFiles/glove.6B.300d.txt',outputPath = '../WordVecFiles/'):
	# Add 1 for <eos> token
	vocabSize = len(word_tokens) + 2

	print 'Loading pretrained vectors from file...'
	globalWordTokens, globalWordVectors = extractVecs(pretrained)
	## Get index of word (in corpus) in the GloVe vector file
	word_dict = {}
	word_ind = 1
	OOV_words = 0
	t0 = time.clock()
	word_vecs = np.zeros((vocabSize,300))
	print 'Assigning gLoVe vectors to work tokens...'
	for word in word_tokens:
		try:
			indValue = np.where(globalWordTokens == word.lower())[0]
			if bool(indValue) is True:
				 word_dict[word] = word_ind	        
				 word_vecs[word_ind,:] = globalWordVectors[indValue[0],:]
			else:
				#print   '"%s" does not appear in the gLoVe Dataset. Assigned random Word Vector' %word
				word_dict[word] = word_ind
				word_vecs[word_ind,:] = np.random.uniform(-0.1,0.1,size = 300)
				OOV_words+=1
			word_ind +=1
			
		except Exception as e:
			print word,'\t', indValue, type(word)
			print e.message
	word_dict['<end>'] = word_ind
	word_dict['<go>'] = word_ind+1
	word_vecs[vocabSize-2] = np.random.uniform(-0.1,0.1,size=300)
	word_vecs[vocabSize-1] = np.random.uniform(-0.1,0.1,size=300)

	print time.clock() - t0, " taken to process the text corpus and assign word vectors. Total of %d OOV tokens out of %d vocabulary size" %(OOV_words,vocabSize)

	with open(outputPath+'wordToInd.dict','wb') as f:
		cPickle.dump(word_dict,f,protocol =2)
	with open(outputPath+'wordVecs.matrix','wb') as f:
		cPickle.dump(word_vecs,f,protocol=2)
	return word_dict,word_vecs
