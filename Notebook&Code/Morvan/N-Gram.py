import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
x_np = np.ones(10)
x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
# x1 = torch.from_numpy(x_np[1, :, 1])
# print(
#     "x_np: ", x_np,
#     "\nx: ", x
# )

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# print(test_sentence)
#  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])
vocab = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)
losses = []
loss_func = nn.NLLLoss()

for epoch in range(100):
    total_loss = 0
    # print(word_to_ix)
    for context, target in trigrams:
        # print(context)
        # print(target)
        # 因为通过了set去重，word_to_ix 对应的是key：value是word:index，好像也没啥意义
        # Step 1. Prepare the inputs to be passed to the model
        # (i.e, turn the words into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_idx[w]
                                     for w in context], dtype=torch.long)
        # 给出上下文，预测单词
        # 2. zero gradient
        model.zero_grad()
        # 3. forward
        log_probs = model(context_idxs)
        # 预测结果的loss
        # 4. loss
        loss = loss_func(log_probs, torch.tensor(
            [word_to_idx[target]], dtype=torch.long))
        # 5. bp
        loss.backward()
        # 6. next step
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
# torch.save(model, ".")
# print(word_to_ix)
# p = model(torch.tensor([52, 15], dtype=torch.long))
# print(max(p.numpy()))

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

word, label = trigrams[1]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = torch.LongTensor([word_to_idx[i] for i in word])
print("int: {}".format(word))
out = model(word)
pred_label_idx = out.max(1)[1].item()
print("out: ", out)
print("out.max: ", out.max(1))
print("out.max(1)[1]: ", out.max(1)[1])
print("pred_label_idx: ", pred_label_idx)
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
