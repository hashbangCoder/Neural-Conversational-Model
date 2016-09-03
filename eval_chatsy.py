import tensorflow as tf
from tensorflow.python.ops import embedding_ops,rnn_cell,rnn
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import inspect_checkpoint as ic
import cPickle

##Build the graph
def eval_chatsy(options):
    #checkpt = '../Checkpoints/checkpoint_48000'
    vocab_size = 49452
    embedding_dims = 300
    print 'Loading saved glove embeddings for tokens...'
    with open(options.wvec_dict,'rb') as f:
        word_dict = cPickle.load(f)	
    with open(options.wvec_mat,'rb') as f:
        word_vecs = cPickle.load(f)
    inv_word_dict = {v: k for k, v in word_dict.iteritems()}
    ## Add <UNK> token
    embedding = tf.Variable(initial_value=tf.convert_to_tensor(word_vecs),name='embed_vecs',trainable=False)

    encoder_inp  = tf.placeholder(tf.int32,(None,))
    encoder_exp = tf.expand_dims(encoder_inp,0)
    decoder_inp = tf.placeholder(tf.int32,(1))
    #decoder_exp = tf.expand_dims(decoder_inp,0)
    state = tf.placeholder(tf.float32,(1024,))
    state_exp = tf.expand_dims(state,0)

    encoder_batch = tf.cast(embedding_ops.embedding_lookup(embedding,encoder_exp),tf.float32)
    decoder_batch = tf.cast(embedding_ops.embedding_lookup(embedding,decoder_inp),tf.float32)

    print('Chatsy is now online and you can start interacting...\n')
    inp = raw_input('You:  ')

    # TODO: Add support for unknown input tokens. Expand vocabulary
    inp_vector =np.array([word_dict[item] for item in wordpunct_tokenize(inp.strip().lower())])

    sess = tf.InteractiveSession()
    init_op = tf.initialize_variables([embedding])
    init_op.run()
    with tf.variable_scope('seqToseq') as scope:
        with tf.variable_scope('enc'):
            gru_cell = rnn_cell.GRUCell(1024)
            _,encoder_state = rnn.dynamic_rnn(rnn_cell.GRUCell(1024),encoder_batch,dtype = tf.float32)
        with tf.variable_scope('dec/RNN'):
            with tf.variable_scope('gru_dec'):
                gru_dec = rnn_cell.GRUCell(1024)
            output,dec_state = gru_dec(decoder_batch,state_exp)
        #output1,state1 = rnn.dynamic_rnn(gru_cell,tf.expand_dims(decoder_batch,0),dtype=tf.float32,time_major=False)
        W = tf.get_variable('linear_W',[vocab_size,1024],dtype = tf.float32)
        b = tf.get_variable('linear_b',[vocab_size],dtype= tf.float32)
        #This part adapted from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
        word_ind = tf.argmax(tf.nn.softmax(tf.matmul(tf.reshape(output,[-1,1024]),tf.transpose(W)) + b),1)

    saver =tf.train.Saver()
    saver.restore(sess,options.load_checkpt)
    enc_state = encoder_state.eval(feed_dict = {encoder_inp:inp_vector})
    exit
    stopGenFlag = False
    token_count = 1
    output = []
    while not stopGenFlag:
        if token_count == 1:
            dec_inp = np.array([word_dict['<go>']])
        out_token_ind,new_state = sess.run([word_ind,dec_state],feed_dict={state:enc_state.squeeze(),decoder_inp:dec_inp})
        output.append(inv_word_dict[out_token_ind[0]])
        assert out_token_ind <= vocab_size, 'unrecognized token'
        dec_inp = out_token_ind
        enc_state = new_state
        token_count +=1
        if output[-1] == '<end>':
            stopGenFlag = True
        if token_count > 10:
            break
    print 'Chatsy:  ', ' '.join(output[:-1])
#ic.print_tensors_in_checkpoint_file('../Checkpoints/checkpoint_48000',None)  

def main_eval(chkpt):
	eval_chatsy(chkpt)
	while raw_input('\n\nTalk to Chatsy again? : ').strip().lower() == 'y':
		print '\n\n'
		tf.reset_default_graph()
		eval_chatsy(chkpt)
