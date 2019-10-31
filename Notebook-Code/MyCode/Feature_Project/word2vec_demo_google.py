import word2vec
# 1. 数据预处理
# 有bug,只能用命令行来
# word2vec.word2phrase('/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8',
#                            '/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8-phrases', verbose=True)

# That created a text8.bin file containing the word vectors in a binary format. the size of word vector is 100
# word2vec.word2vec('/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8-phrases',
#                   '/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8.bin', size=100, verbose=True)

# That created a text8-clusters.txt with the cluster for every word in the vocabulary
# word2vec.word2clusters('/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8',
#                        '/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8-clusters.txt', 100, verbose=True)

# 2. 网络模型训练
model = word2vec.load(
    '/Users/hwolf/Documents/ML/Task/MyCode/Feature_Project/text8.bin')

print('model.vocab', model.vocab)
print('model.vectors.shape', model.vectors.shape)
print('model.vectors', model.vectors)

print('model[\'dog\'][: 10]', model['dog'][:10])

print(model.distance("dog", "cat", "fish"))
