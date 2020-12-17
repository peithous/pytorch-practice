import torch 
import torchtext
from torchtext.vocab import Vectors, GloVe
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="conv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field()
LABEL = data.Field(sequential=False, unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train, val, test, vectors='glove.6B.100d')
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

VOCAB_SIZE = len(TEXT.vocab)
CLASSES = len(LABEL.vocab) 
EMB_SIZE = TEXT.vocab.vectors.shape[1]

#padded conv
#dropout on linear
#weight_dec instead of l2 norm
class CnnRegression(nn.Module):  
    def __init__(self, num_labels, vocab_size, emb_size):
        super(CnnRegression, self).__init__()

        self.emb = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True)
        self.conv1 = nn.Conv1d(emb_size, emb_size, 3, padding=2) # [input, filter, kernel, pad]
        self.conv2 = nn.Conv1d(emb_size, emb_size, 4, padding=2)
        self.conv3 = nn.Conv1d(emb_size, emb_size, 5, padding=4)
        self.linear = nn.Linear(emb_size*3, num_labels)
        self.dropout = nn.Dropout(0.25)

    def forward(self, batch_input):
        #print('b', batch_input.shape)
        e = self.emb(batch_input) # [batch_size, batch_seqlen, emb_size]
        e = torch.transpose(e, 1, 2) # [batch_size, emb_size, batch_seqlen]
        #print('e0', e.shape) 
        
        e1 = self.conv1(e) # [batch_size, emb_size, batch_seqlen-kernel_size+1]
        #print('e', e.shape)
        e1 = torch.max(e1, 2)[0] # [batch_size, emb_size] 
        # torch.max returns tuple (values, index)
        #print('e1', e1.shape)

        e2 = self.conv2(e)
        e2 = torch.max(e2, 2)[0]
        #print('e2', e2.shape)

        e3 = self.conv3(e)
        e3 = torch.max(e3, 2)[0]
        #print('e3', e3.shape)

        e = torch.cat((e1, e2, e3), dim=1)
        e = F.relu(e)
        #print('e', e.shape)
        #print('l', self.linear(e).shape)

        e = self.dropout(e)
        
        probs = F.log_softmax(self.linear(e), dim=1)
        #print('probs.shape', probs.shape)
        return probs

def make_batch_token_index(batch_text):
    blen = batch_text.shape[1]
    seqlen = batch_text.shape[0]
    
    vecs = []
    for b in range(blen):
        vec_b = []
        for s in range(seqlen):
            w = batch_text[s, b].item()
            vec_b.append(w)
        vecs.append(vec_b)

    assert len(vecs[0]) == seqlen
    vecs = torch.LongTensor(vecs)
    #print(vecs)
    return vecs

model = CnnRegression(CLASSES, VOCAB_SIZE, EMB_SIZE).to(device)

loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.1) # no dropout is better, ~79% 
optimizer = optim.Adadelta(model.parameters(), lr=0.15, weight_decay=0.01) # dropout = 0.25 
    # lr=0.1 seems to have lower variace w/out reg, and test acc is ~80.05% 
    # lr=0.15 gets over 81.5% and usually above 80% 
    # weight_decay does not help

for epoch in range(50):
    for batch in train_iter:
        model.zero_grad()
        batch_emb_matrix = make_batch_token_index(batch.text)
        target = (batch.label).to(device)
        #print('target.shape', target.shape)

        log_probs = model(batch_emb_matrix).to(device)
        #print('log_probs', log_probs)
        #print('s', log_probs.shape)

        loss = loss_function(log_probs, target)
        
        writer.add_scalar('loss', loss, epoch)

        loss.backward()
        #print(epoch, loss)
        optimizer.step()

model.eval()

total_error = []
for batch in test_iter:
        #print(batch.text.shape)

        bow_vector = make_batch_token_index(batch.text)
        target = torch.LongTensor(batch.label).to(device)

        log_probs = model(bow_vector).to(device)
        
        _, argmax = log_probs.max(1)
        #print(log_probs.shape)
        #print(log_probs.max(1))

        error = torch.abs(argmax - batch.label)
        error = sum(error)
        error = error.item()/len(batch.label) 
        #print(error)
        writer.add_scalar('error', error, epoch)

        total_error.append(error)

total_error = sum(total_error)/len(total_error)
print('test error: ', total_error)