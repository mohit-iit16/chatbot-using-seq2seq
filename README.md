# chatbot-using-seq2seq

Overview:

This is an implementation of seq2seq network for designing a chatbot. This model is trained on [Microsoft Research Social Media Conversation Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52375&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F6096d3da-0c3b-42fa-a480-646929aa06f1%2F) 
Dataset consist of a series of tweet Ids which form a dialog between two people. Data needs to be manupulated to be made fit for feeding to the model. After few hours of training, chatbot can hold an interesting conversation.

Dependencies:
1. Numpy
2. six
3. nltk (for data preprocessing)
4. tensorflow (version 1.1.0 will throw an error which they are going to fix in next release. Use version 1.0.0 instead.)


Other datasets that can also be used to train:
1. Ubuntu dialog corpus v2.0 (https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
2. Your own chat data
