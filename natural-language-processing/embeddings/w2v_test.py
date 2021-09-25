# # Jon's Pseudocode
# # -------------------------
# def input_layer(word_idx):
# 	x = torch.zeros(voca_size)
# 	x[word_idx] = 1.0
# 	return x

	
# def train(n_epochs= INT, lr = Float, embedding_size = INT):

#     W1 = Variable( torch.random(vocab_size, embedding_size).float(), requires_grad=True )
#     W2 = Variable( torch.random(embedding_size, vocab_size).float(), requires_grad=True)
    
#     for epoch in epochs:
    
#         loss_val = 0
        
#         for data, target in dataset:
        
#             x = variable(input_layer(data)).float
#             y_true = Variable(torch.numpy(np.array([target])).long())
            
#             z1 = matmul(x,W1)
#             z2= matmul(z1,W2)
            
#             log_softmax = log_softmax(z2, dim0)
#             loss = NLLloss(log_softmax(1,-1), y_true)
            
#             loss_val += loss
            
#             W1.data -= lr * W1.gradient_data
#     W2.data -= lr * W2.gradient_data
    
#     W1.gradient_data = 0
#     W2.gradient_data = 0
    
#     if epoch % 10 == 0:    
#         print(f'Loss at epoch {epoch}: {loss_val/len(dataset)}')
# # -------------------------


# # GRU
# # -------------------------
# class CommandScorer(nn.Module):
# 	def __init__(self, input_size, hidden_size):
#         	super(CommandScorer, self).__init__()
#         	...
# 					self.embedding    = nn.Embedding(input_size, hidden_size)
# 					self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
# 					...
# 	def forward(self, obs, commands, **kwargs):
# 					embedded = self.embedding(obs)
#         	_, encoder_hidden = self.encoder_gru(embedded)
#         	...
# # -------------------------
# GRU - No idea what this is
# class CommandScorer(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.embedding    = nn.Embedding(input_size, hidden_size)
#         self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
	
#     def forward(self, obs, commands, **kwargs):
#         embedded = self.embedding(obs)
#         _, encoder_hidden = self.encoder_gru(embedded)


#########################
import numpy as np
import torch
import torch.nn as nn
from prepro_from_scratch import pair_pipeline

dataset, vocab_size, w2i, i2w = pair_pipeline("natural-language-processing/data/poems.txt")
# print(dataset[0])
# for data, target in dataset:
#     print(data, "  -  ", target)
#     break

# One hot encode the input word
def input_layer(word_index):
    x = torch.zeros(vocab_size)
    x[word_index] = 1.0
    return x


def train(n_epochs:int, lr:float, embedding_size:int):
    weights_1 = torch.rand(vocab_size, embedding_size, dtype=torch.float, requires_grad=True)
    weights_2 = torch.rand(embedding_size, vocab_size, dtype=torch.float, requires_grad=True)
    
    for epoch in range(n_epochs):
        lossval = 0

        for data, target in dataset:

            x = torch.Tensor(input_layer(data))
            # y_ndarry = target.as_type(ndarray)

            # Correction: Convert this to a long
            y_true = torch.from_numpy(np.array([target])).long()




            z1 = torch.matmul(x, weights_1)
            z2 = torch.matmul(z1, weights_2)


            ### log_softmax = log_softmax(z2, dim0)
            LS = nn.LogSoftmax(dim=0)
            my_log_softmax = LS(z2)

            ### loss = nn.NLLloss(log_softmax(1,-1), y_true)
            loss_f = nn.NLLLoss()
            # Correction: use .view()
            loss = loss_f(my_log_softmax.view(1,-1), y_true)
            # loss = nn.NLLLoss(log_softmax, y_true)

            
            loss.backward()
            lossval += loss

            # Correction: Use X.grad.data
            weights_1.data -= lr * weights_1.grad.data
            weights_2.data -= lr * weights_2.grad.data
            # Correction: Use .zero_() to set to zero
            weights_1.grad.data.zero_()
            weights_2.grad.data.zero_()

            if epoch % 1000 == 0:
                print(f"Loss at epoch {epoch + 1}: {lossval/len(dataset)}")


    return weights_1

if __name__ == "__main__":
    # Embedding size should be something like 256+ depending on hardware
    result = train(100, 0.01, 256)
    print(result)
