from optparse import OptionParser
import datetime
import utils,loadWordVecs
import os,random,cPickle,sys
from tensorflow.python.ops import embedding_ops,rnn_cell,rnn
import tensorflow as tf

parser = OptionParser()
parser.add_option("-i", "--input-path", dest="input_path",help="Path to data text files",default = '../../Conversational Data/cornell movie-dialogs corpus/TFRecordFormat/')  
parser.add_option("-p", "--pretrained", dest="pretrained",help="Weights file path",default = '../saved_files/wordEmbedMat.pkl')
parser.add_option("-v", "--vector-dict", dest="vector_dict",help="Word TO Index Dictionary mapping",default = '../WordVecFiles/wordToIndex.dict')
parser.add_option("-w", "--output-weights", dest="outputWeights",help="Weights output file name",default = '../Output/ModelParams/LSTM_params_finetuned.h5')
parser.add_option("-s", "--split-ratio", dest="split",help="Train data percentage",default = 0.8)
parser.add_option("-b", "--batch-size", dest="batch_size",help="Size of mini batch",default = 128)

parser.add_option("--tboard-dir", dest="tboard_dir",help="Directory to log tensorfboard events",default = './Summaries/')
parser.add_option("--save-path", dest="save_path",help="Path to save checkpoint",default = '../Checkpoints/')
parser.add_option("--save-freq", dest="save_freq",help="Frequency with which to save checkpoint",default = 10000)
parser.add_option("-r", "--learning-rate", dest="lr",help="Learning Rate",default = 0.001)
parser.add_option("-e", "--num-epochs", dest="num_epochs",help="Number of epochs",default = 20)
#parser.add_option("-R", "--runs", dest="runs",help="Number of runs to average results over",default = 1)
parser.add_option("--neurons", dest="neurons",help="Neurons in each layer",default = '128,64')
(options, args) = parser.parse_args()

fileList = os.listdir(options.input_path)
if fileList == []:
	print '\nNo TFRecord file found. Saving Text Files as TFRecords for easy processing at : %s' %options.input_path,'\nThis will take a few minutes...\n\n'
	word_dict,word_vecs = utils.saveDataAsRecord()
else :
	try:
		print 'Loading saved glove embeddings for tokens...'
		with open('../WordVecFiles/wordToInd.dict','rb') as f:
			word_dict = cPickle.load(f)	
		with open('../WordVecFiles/wordVecs.matrix','rb') as f:
			word_vecs = cPickle.load(f)
	except IOError:
		print '[ERROR]gLoVe Vecotrs and Dictionary Files not found'
		sys.exit(0)
#trainFiles,testFiles = fileList[:int(float(options.split)*len(fileList))],fileList[int(float(options.split)*len(fileList)):]
inv_word_dict = {v: k for k, v in word_dict.iteritems()}
inv_word_dict[0] = '' #Add padding-decode for completeness
dataQAlive = True
vocab_size = len(word_dict)
batch_size = int(options.batch_size)
fileList = [options.input_path+item for item in fileList]

# Proto for parsing the TFRecord	
lengths_context = {'utter_length':tf.FixedLenFeature([], dtype=tf.int64),'resp_length':tf.FixedLenFeature([], dtype=tf.int64)}
convo_pair = {"utterance": tf.FixedLenSequenceFeature([], dtype=tf.int64),"response": tf.FixedLenSequenceFeature([], dtype=tf.int64),"labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
embedding = tf.Variable(tf.python.ops.convert_to_tensor(word_vecs),name='embed_vecs')

trainQ = tf.train.string_input_producer(fileList)
RecReader = tf.TFRecordReader()

batch_strings = RecReader.read(trainQ)
con,seq=tf.parse_single_sequence_example(batch_strings.value,context_features=lengths_context,sequence_features=convo_pair,name='parse_ex')
encoder_inputs,decoder_inputs,labels_out,enc_len,dec_len = seq['utterance'],seq['response'],seq['labels'],con['utter_length'],con['resp_length']
#decoder_inputs = decoder_inputs[:tf.shape(decoder_inputs)[0]-1]
#dec_len -= -1
mini_batch = tf.train.batch([encoder_inputs,decoder_inputs,enc_len,dec_len,labels_out],batch_size,2,capacity=50*batch_size,dynamic_pad = True,enqueue_many=False)
encoder_inp,decoder_inp,encoder_lens,decoder_lens,labels = mini_batch

# Decrease lengths by 1 to account for <eos> in decoder_inp i.e. to align labels and logits. Remove first element of labels
#encoder_lens = tf.sub(encoder_lens,1)
#decoder_lens = tf.sub(decoder_lens,1)
encoder_batch =tf.cast( embedding_ops.embedding_lookup(embedding,mini_batch[0]),tf.float32)
decoder_batch = tf.cast(embedding_ops.embedding_lookup(embedding,mini_batch[1]),tf.float32)


## Build graph for seq-seq model
print '\n'*2,'Building Seq-Seq Graph...'
with tf.variable_scope('seqToseq') as scope:
	gru_cell = rnn_cell.GRUCell(1024)
	_,encoder_state = rnn.dynamic_rnn(rnn_cell.GRUCell(1024),encoder_batch,sequence_length=encoder_lens,dtype = tf.float32)
	with tf.variable_scope('dec'):
		output,_ = rnn.dynamic_rnn(gru_cell,decoder_batch,initial_state = encoder_state,
											 sequence_length = decoder_lens,time_major=False)
	W = tf.get_variable('linear_W',[vocab_size,1024],dtype = tf.float32)
	b = tf.get_variable('linear_b',[vocab_size],dtype= tf.float32)
	#This part adapted from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
	logits_flat = tf.matmul(tf.reshape(output,[-1,1024]),tf.transpose(W)) + b
	labels_flat = tf.reshape(labels,[-1])
	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, labels_flat)
	masked_losses = tf.reshape(tf.sign(tf.to_float(labels_flat)) * losses,tf.shape(labels))
	mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(decoder_lens,tf.float32)
	mean_loss = tf.reduce_mean(mean_loss_by_example)
	tf.scalar_summary('summary/batch_loss' , mean_loss)
	optim = tf.train.AdamOptimizer()
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optim.minimize(mean_loss,global_step = global_step)

#label_batch = tf.train.batch(labels,batch_size,2,capacity=3*batch_size+1,dynamic_pad = True,enqueue_many=True
merged = tf.merge_all_summaries()
saver = tf.train.Saver()

#with tf.Session() as sess:
sess = tf.Session()
coord = tf.train.Coordinator()
init_op = tf.initialize_all_variables()
threads = tf.train.start_queue_runners(coord=coord,sess = sess)
sum_writer = tf.train.SummaryWriter(options.tboard_dir, graph=sess.graph)
sess.run(init_op)
print 'Start Training...'
try:
	while not coord.should_stop():
		_,batch_loss,train_step,summary,dec_lens = sess.run([train_op,mean_loss,global_step,merged,decoder_lens])
		if train_step%100:
			sum_writer.add_summary(summary,train_step)
			print 'Mini-Batches run : %d\t\tLoss : %f' %(train_step,batch_loss)
			print dec_lens
		if coord.should_stop():
			print 'wut'
			break
		if train_step%int(options.save_freq)==0:
			saver.save(sess,options.save_path+'checkpoint_'+str(train_step))
			print '@iter:%d \t Model saved at:'%(train_step,options.save_path)

except tf.errors.OutOfRangeError:
		dataQAlive = False
		print 'Training Done..'
finally:
		coord.request_stop()
		coord.join(threads)



