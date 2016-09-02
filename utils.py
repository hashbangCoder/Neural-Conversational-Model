import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os,sys,random,tempfile
from nltk.tokenize import wordpunct_tokenize
import loadWordVecs,ast

def getWordList(filePath='../../Conversational Data/cornell movie-dialogs corpus/movie_lines.txt',dataset = 'Cornell DB'):
	if dataset == 'OSD':
		fileList = os.listdir(filePath)
		data = []
		for fileName in fileList:
			if os.path.isfile(filePath+fileName):
				with open(filePath+fileName,'r') as f:
					data.append(f.read().lower())
		return list(set(wordpunct_tokenize('\n'.join(data))))
	elif dataset == 'Cornell DB':
		with open(filePath,'r') as f:
			full_text = f.readlines()
		data = []
		word_list = []
		print 'Getting word list and reading data from file...'
		for line in full_text:
			split_line = line.split('+++$+++')
			data.append((split_line[-1].strip().lower(),int(split_line[0][1:].strip())))
			word_list.append(split_line[-1].strip().lower())
		word_list = list(set(wordpunct_tokenize('\n'.join(word_list))))
		return data,word_list

def createSample(utter,resp):
## Adapted from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
	example = tf.train.SequenceExample()
	utterance = example.feature_lists.feature_list["utterance"]
	response = example.feature_lists.feature_list["response"]
	labels = example.feature_lists.feature_list["labels"]
	#TF's lack of variable/negative indexing means labels have to stored separately as well
	labs = resp[1:]
	resp = resp[:-1]
	utter = utter[1:]
	example.context.feature["utter_length"].int64_list.value.append(len(utter))
	example.context.feature["resp_length"].int64_list.value.append(len(resp))
	for item in utter:
		utterance.feature.add().int64_list.value.append(item)
	for item in resp:
		response.feature.add().int64_list.value.append(item)
	for item in labs:
		labels.feature.add().int64_list.value.append(item)
	#utterance.feature.add().int64_list.value.extend(utter)
	#response.feature.add().int64_list.value.extend(resp)
	return example


def saveDataAsRecord(wordToIndDict = '../WordVecFiles/wordToIndex.dict',dataset = 'Cornell DB',filePath = '../../Conversational Data/cornell movie-dialogs corpus/'):
	if dataset == 'OSD':
		fileList = os.listdir(filePath)
		fileList = [filePath+item for item in fileList]
		for _file in FileList: 
			#Two sets of conversations for each file
			for ind in [0,1]:
				with open(filePath+'TFRecordFormat/'+str(ind)+_file,'wb') as f:
					writer = tf.python_io.TFRecordWriter(f)
					with open(_file,'r') as f1:
						text = f1.readlines() if ind==0 else f1.readlines()[1:]
					for val in range(0,len(text),2):
						example = createSample(text[val],text[val+1],word_dict)
						writer.write(example.SerializeToString())
					writer.close()
	elif dataset == 'Cornell DB':
		data,wordList = getWordList(dataset = 'Cornell DB',filePath = '../../Conversational Data/cornell movie-dialogs corpus/movie_lines.txt')
		word_dict,word_vecs = loadWordVecs.generateWordDict(wordList)
		del wordList
		dialogVector = {} 
		print 'Creating line# - Dialogue Seq. mapping due to format of data...'
		for sample in data:
			dialogVector[sample[1]] = [word_dict[item] for item in ['<go>']+wordpunct_tokenize(sample[0])+['<end>']]
		with open(filePath+'movie_conversations.txt','r') as f:
			full_convo = f.readlines() 
		writer = tf.python_io.TFRecordWriter(filePath+'TFRecordFormat/'+'CornellMovieDialogueCorpusTF.record')
		print 'Converting string to vector representation and storing as TFRecords...'
		discard_convo = 0
		for convo in full_convo:
			convo_data = [dialogVector[int(item[1:])] for item in ast.literal_eval(convo.split('+++$+++')[-1].strip())]
			for ind in range(len(convo_data)-1):
				if len(convo_data[ind]) ==1 or len(convo_data[ind+1]) == 1:
					discard_convo+=1
					continue
				example = createSample(convo_data[ind],convo_data[ind+1])
				writer.write(example.SerializeToString())
		print 'Discarded total of %d conversation parirs due to empty utterance/response in dataset' %discard_convo
		writer.close()
		print 'Done. Files stored at : ',filePath+'TFRecordFormat/'+'CornellMovieDialogueCorpusTF.record'
		return #word_dict,word_vecs


