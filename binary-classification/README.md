### Binary classification
Binary classification on torchtext.datasets.SST
- https://pytorch.org/text/_modules/torchtext/datasets/sst.html
- https://nlp.stanford.edu/sentiment/

Test set accuracies on default splits, evaluated on default train-test split without cross validation:

#### Naive Bayes: 
~83% acc
	- Wang and Manning (https://www.aclweb.org/anthology/P12-2018.pdf) have 83.45% acc on RT-2k with MNB-uni 

#### Logistic regression with bag of words features: 
~78% acc

#### Logisitc regression with features from pretrained glove.6B.100d: 
~77% acc

#### Logisitc regression with features from CNNs as in Kim 2014 (without phrase filters): 
- \>80->81.5% acc with Adadelta, dropout_rate = 0.25, weight_decay = 0
- similarly with weigh_decay = 0.01
	- Kim 2014 has >81% acc (https://github.com/yoonkim/CNN_sentence/blob/master/README.md) with Adadelta, dropout = 0.5 and l2 regularization
	- more on dropout and l2: https://arxiv.org/pdf/1510.03820.pdf