# Neural-Conversational-Model
Tensorflow Implementation of Neural Conversational Model by Vinyals et.al. - http://arxiv.org/pdf/1506.05869v3.pdf

Includes optimizations like TFRecords, dynamic_rnn implementation and pretrained word-embeddings to speed up training. Trained on the excellently formated Cornell Movie Dialogure corpus available at - http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html


####PRE-REQUISITES:
1. Tensorflow

1. CUDNN, CUDA 

1. Python's NLTK package

1. Stanford's Pretrained GloVe Vectors available [here](http://nlp.stanford.edu/projects/glove/). Code currently only supports 300-dim vectors

1. Cornell's Movie Dialog dataset



####HOW TO RECREATE :
Requires a Tensorflow supported GPU. CPU version not supported. With all pre-requisites installed

1.  In `loadWordVecs.py` set paths to downloaded GloVe file and output path in function `generateWordDict`, 

1. `CUDA_VISIBLE_DEVICES=1 python chatbot.py`



####TO EVALUATE (Requires a checkpoint file): 
NOTE : Currently out of vocabulary words are not supported. So for some inputs (some proper nouns, uncommon words) it will throw an error. Will add support soon.
`CUDA_VISIBLE_DEVICES=1 python eval_chatsy.py --run-mode 1 --load-chkpt <path to valid checkpoint file>`



####EXAMPLE CONVERSATIONS AFTER ROUGHLY 8 EPOCHS : 

Convo 1:
>User : Hi!

>Chatty: Hi, Charlie

Convo 2:
>User : How are you?

>Chatty : I'm not sure.

Convo 3:
>User : Are you alive?

>Chatty : I'm not going to hurt you.

>User : ...  (quietly backs out of chat)

NOTE : The example conversations were slightly edited to correct misspellings (ex: "I' m") in the training corpus hence the learnt vocabulary. But otherwise unaltered in any way.

