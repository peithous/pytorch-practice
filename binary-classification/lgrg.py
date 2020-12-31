import torch 
import torchtext
from torchtext.vocab import Vectors, GloVe
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logreg")

TEXT = data.Field()
LABEL = data.Field(sequential=False, unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
# does not work without test and val (19423 vs 16282) 
#print(len(vars(TEXT.vocab)['freqs']))
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

VOCAB_SIZE = len(TEXT.vocab)
CLASSES = len(LABEL.vocab) 

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        blen = bow_vec.shape[1]

        probs = []

        for b in range(blen):
            #print('v', bow_vec[:, b].view(1, -1).shape)
            #print('l', self.linear(bow_vec[:, b].view(1, -1)).shape)
            #print('sf', F.log_softmax(self.linear(bow_vec[:, b].view(1, -1)), dim=1).shape)
            probs.append(F.log_softmax(self.linear(bow_vec[:, b].view(1, -1)), dim=1))
                    
        probs = torch.squeeze(torch.stack(probs))
        #print(probs)
        return probs

def make_bow_vector(batch_text):
    blen = batch_text.shape[1]
    seqlen = batch_text.shape[0]

    vec = torch.zeros(VOCAB_SIZE, blen)
    
    for b in range(blen):
        for s in range(seqlen):
            w = batch_text[s, b]           
            vec[w.item(), b] += 1

    #print(vec.shape)
    return vec

model = BoWClassifier(CLASSES, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):

    for batch in train_iter:
        model.zero_grad()
        bow_vector = make_bow_vector(batch.text)
        target = batch.label
        #print(target.shape)

        log_probs = model(bow_vector)
        #print(log_probs)
        #print(log_probs.shape)

        loss = loss_function(log_probs, target)
        
        writer.add_scalar('loss', loss, epoch)

        loss.backward()
        #print(epoch, loss)
        optimizer.step()
        #print(loss)
        #print(log_probs.shape)
        

total_error = []
for batch in test_iter:
        bow_vector = make_bow_vector(batch.text)
        target = torch.LongTensor(batch.label)
        #print(bow_vector.shape)

        log_probs = model(bow_vector)
        #print(log_probs.max(1))
        if batch.label.shape[0] != 1:
            _, argmax = log_probs.max(1)
        #print(log_probs.shape)
        #print(log_probs.max(1))

        else:
            argmax = torch.argmax(log_probs)

        error = torch.abs(argmax - batch.label)
        error = sum(error)
        error = error.item()/len(batch.label) 
        #print(error)
        #writer.add_scalar('error', error, epoch)

        total_error.append(error)

total_error = sum(total_error)/len(total_error)
print('test error: ', total_error)