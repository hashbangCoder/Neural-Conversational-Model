# Neural-Conversational-Model
Tensorflow Implementation of Neural Conversational Model by Vinyals et.al. - http://arxiv.org/pdf/1506.05869v3.pdf

Uses TFRecords, dynamic_rnn implementation and pretrained word-embeddings to speed up training. Trained on the excellently formated Cornell Movie Dialogure corpus - http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html


###After training for 48,000 mini-batches (roughly 8 epochs), some of the outputs : 
Convo 1:
>User : Hi!

>Chatty: Hi, Charlie

Convo 2:
>User : How are you?

>Chatty : I'm not sure.

Convo 3:
>User : Are you alive?

>Chatty : I'm not going to hurt you.

This is not what I inputed but *definitely what I felt*
>User : ...  (quietly backs out of chat)

NOTE : The outputs are slightly formatted to remove the <eos> token and to correct misspellings (ex: "I' m") in the training corpus hence the vocabulary. But otherwise it has not been altered in any way

I'll update how to run the code and format it a bit more soon!
