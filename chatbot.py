from optparse import OptionParser
import utils,loadWordVecs
import os,random,cPickle,sys
from tensorflow.python.ops import embedding_ops,rnn_cell,rnn
import tensorflow as tf
from numpy import isnan
import eval_chatsy

parser = OptionParser()
parser.add_option("-i", "--input-path", dest="input_path",help="Path to data text files in TFRecord format",
										default = '../../Conversational Data/cornell movie-dialogs corpus/TFRecordFormat/')  
parser.add_option("--wordvec-dict", dest="wvec_dict",help="Path to saved word-index dictionary",default = '../WordVecFiles/wordToInd.dict')
parser.add_option("--wordvec-mat", dest="wvec_mat",help="Path to saved index-wordvector numpy matrix ",default = '../WordVecFiles/wordVecs.matrix')
parser.add_option("-b", "--batch-size", dest="batch_size",help="Size of mini batch",default = 32)

parser.add_option("--tboard-dir", dest="tboard_dir",help="Directory to log tensorfboard events",default = './Summaries/')
parser.add_option("--save-path", dest="save_path",help="Path to save checkpoint",default = '../Checkpoints/')
parser.add_option("--save-freq", dest="save_freq",help="Frequency with which to save checkpoint",default = 2000)
parser.add_option("-r", "--learning-rate", dest="lr",help="Learning Rate",default = 0.001)
parser.add_option("-e", "--num-epochs", dest="num_epochs",help="Number of epochs",default = 20)

parser.add_option("--run-mode", dest="mode",help="0 for train, 1 for test",default = 0 )
parser.add_option("--load-chkpt", dest="load_chkpt",help="Path to checkpoint file. Required for mode:1",default = '')
(options,_) = parser.parse_args()


def train(options):
	fileList = os.listdir(options.input_path)
	if fileList == []:
		print '\nNo TFRecord file found. Saving Text Files as TFRecords for easy processing at : %s' %options.input_path,'\nThis will take a couple minutes...\n\n'
		word_dict,word_vecs = utils.saveDataAsRecord()
	else :
		try:
			print 'Loading saved glove embeddings for tokens...'
			with open(options.wvec_dict,'rb') as f:
				word_dict = cPickle.load(f)	
			with open(options.wvec_mat,'rb') as f:
				word_vecs = cPickle.load(f)
		except IOError:
			raise Exception('[ERROR]gLoVe Vectors and Dictionary Files not found')

	inv_word_dict = {v: k for k, v in word_dict.iteritems()}
	inv_word_dict[0] = '' #Add padding-decode for completeness
	vocab_size = len(word_dict)
	batch_size = int(options.batch_size)
	fileList = [options.input_path+item for item in fileList]

	# Proto for parsing the TFRecord	
	lengths_context = {'utter_length':tf.FixedLenFeature([], dtype=tf.int64),'resp_length':tf.FixedLenFeature([], dtype=tf.int64)}
	convo_pair = {"utterance": tf.FixedLenSequenceFeature([], dtype=tf.int64),
				"response": tf.FixedLenSequenceFeature([], dtype=tf.int64),"labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
	embedding = tf.Variable(tf.python.ops.convert_to_tensor(word_vecs),name='embed_vecs')

	trainQ = tf.train.string_input_producer(fileList)
	RecReader = tf.TFRecordReader()

	batch_strings = RecReader.read(trainQ)
	con,seq=tf.parse_single_sequence_example(batch_strings.value,context_features=lengths_context,sequence_features=convo_pair,name='parse_ex')
	encoder_inputs,decoder_inputs,labels_out,enc_len,dec_len = seq['utterance'],seq['response'],seq['labels'],con['utter_length'],con['resp_length']
	encoder_inputs_rev = tf.reverse(tf.cast(encoder_inputs,tf.int32),[True])
	mini_batch = tf.train.batch([encoder_inputs_rev,decoder_inputs,enc_len,dec_len,labels_out],batch_size
												,num_threads=2,capacity=100*batch_size,dynamic_pad = True,enqueue_many=False)
	encoder_lens,decoder_lens,labels = mini_batch[2:]
	encoder_batch =tf.cast(embedding_ops.embedding_lookup(embedding,mini_batch[0]),tf.float32)
	decoder_batch = tf.cast(embedding_ops.embedding_lookup(embedding,mini_batch[1]),tf.float32)

	## Build graph for seq-seq model
	print '\n'*2,'Building Sequence-Sequence Graph Model...'
	with tf.variable_scope('seqToseq') as scope:
		with tf.variable_scope('enc'):
			gru_cell_enc = rnn_cell.GRUCell(1024)
			_,encoder_state = rnn.dynamic_rnn(gru_cell_enc,encoder_batch,sequence_length=encoder_lens,dtype = tf.float32)
		with tf.variable_scope('dec'):
			with tf.variable_scope('gru_dec'):
				gru_cell_dec = rnn_cell.GRUCell(1024)
			output,_ = rnn.dynamic_rnn(gru_cell_dec,decoder_batch,initial_state = encoder_state,
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

	merged = tf.merge_all_summaries()
	saver = tf.train.Saver()

	sess = tf.InteractiveSession()
	coord = tf.train.Coordinator()
	init_op = tf.initialize_all_variables()
	threads = tf.train.start_queue_runners(coord=coord,sess = sess)
	sum_writer = tf.train.SummaryWriter(options.tboard_dir, graph=sess.graph)
	if options.load_chkpt:
		print 'Loading saved variables from checkpoint file to graph'
		saver.restore(sess,options.load_chkpt)
		print 'Resume Training...'
	else:
		sess.run(init_op)
		print 'Start Training...'
	try:
		saver.save(sess,options.save_path+'checkpoint_start')
		while not coord.should_stop():
			_,batch_loss,train_step,summary= sess.run([train_op,mean_loss,global_step,merged])
			if train_step%100==0:
				sum_writer.add_summary(summary,train_step)
				print '[size:%d]Mini-Batches run : %d\t\tLoss : %f' %(int(options.batch_size),train_step,batch_loss)
			if train_step%int(options.save_freq)==0:
				saver.save(sess,options.save_path+'checkpoint_'+str(train_step))
				print '@iter:%d \t Model saved at: %s'%(train_step,options.save_path)
	except tf.errors.OutOfRangeError:
		print 'Training Complete...'
	finally:
		print 'Saving final checkpoint...Model saved at :',options.save_path
		saver.save(sess,options.save_path+'checkpoint_end')
		print 'Halting Queues and Threads'
		coord.request_stop()
		coord.join(threads)
		sess.close()


if int(options.mode) == 1 and options.load_chkpt is '':
	raise Exception('Require saved checkpoint file for test mode')
elif int(options.mode) == 1 and os.path.isfile(options.load_chkpt):
	eval_chatsy.main_eval(options)
	sys.exit(0)
elif int(options.mode) == 0:
	train(options)
else:
	Exception('Invalid options')


