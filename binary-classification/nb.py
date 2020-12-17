import torch 
import torchtext
from torchtext.vocab import Vectors, GloVe
import math
from torchtext import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT_pos = data.Field()
TEXT_neg = data.Field()
TEXT = data.Field()
LABEL = data.Field(sequential=False, unk_token=None)

train_pos, _, _ = torchtext.datasets.SST.splits(
    TEXT_pos, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral' and ex.label != 'negative')

train_neg, _, _ = torchtext.datasets.SST.splits(
    TEXT_neg, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral' and ex.label != 'positive')

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT_pos.build_vocab(train_pos)
TEXT_neg.build_vocab(train_neg)

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

theta = {}

for x in vars(TEXT.vocab)['freqs']: 
    #print(x, vars(TEXT_pos.vocab)['freqs'][x])
    if x not in theta.keys():
        theta[x] = [1, 1]

for x in vars(TEXT_pos.vocab)['freqs']:
    theta[x][0] += vars(TEXT_pos.vocab)['freqs'][x]

for x in vars(TEXT_neg.vocab)['freqs']:
    theta[x][1] += vars(TEXT_neg.vocab)['freqs'][x]

for x in theta:
    theta[x][0] = math.log(theta[x][0]/(sum(vars(TEXT_pos.vocab)['freqs'].values()) + len(TEXT.vocab)))
    theta[x][1] = math.log(theta[x][1]/(sum(vars(TEXT_neg.vocab)['freqs'].values()) + len(TEXT.vocab)))

theta_index = {}

for x in theta:
    theta_index[vars(TEXT.vocab)['stoi'][x]] = torch.tensor(theta[x])

for x in vars(TEXT.vocab)['stoi'].values():
    if x not in theta_index.keys():
        theta_index[x] = torch.tensor([math.log(1/(sum(vars(TEXT_pos.vocab)['freqs'].values()) + len(TEXT.vocab))), math.log(1/(sum(vars(TEXT_neg.vocab)['freqs'].values()) + len(TEXT.vocab)))])

def model(batch_text):
    blen = batch_text.shape[1]
    seqlen = batch_text.shape[0]

    prior_pos = math.log(vars(LABEL.vocab)['freqs']['positive']/(sum(vars(LABEL.vocab)['freqs'].values())))
    prior_neg = math.log(vars(LABEL.vocab)['freqs']['negative']/(sum(vars(LABEL.vocab)['freqs'].values())))
    
    prior = torch.tensor([prior_pos, prior_neg])
    
    theta_b = []
    for b in range(blen):
        theta_bw = []
        for w in range(seqlen):
            theta_bw.append(theta_index[batch_text[w, b].item()])
        
        theta_bw = torch.stack(theta_bw)
        #print(theta_bw.shape)
        theta_bw = theta_bw.sum(0) + prior
        #print(prior.shape)
        theta_b.append(theta_bw)
    
    theta_b = torch.stack(theta_b)
    
    return theta_b

total_error = []
for batch in test_iter:
    # if batch.label.shape[0] == 1:
    #     print('sth')
    probs = model(batch.text)
    #print(probs.max(1))
    _, argmax = probs.max(1)

    error = torch.abs(argmax - batch.label)
    error = sum(error)
    #print(acc)
    error = error.item()/len(batch.label) 
    total_error.append(error)

total_error = sum(total_error)/len(total_error)
print('test error: ', total_error)