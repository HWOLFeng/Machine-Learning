from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
# docs = ['Well done!',
#         'Good work',
#         'Great effort',
#         'nice work',
#         'Excellent!',
#         'Weak',
#         'Poor effort!',
#         'not good',
#         'poor work',
#         'Could have done better.']
docs = [
    'BGP4MP|1569974454|A|12.0.1.63|7018|99.194.200.0/22|7018 209 22561|IGP|12.0.1.63|0|0|7018:5000 7018:34011|NAG||',
    'BGP4MP|1569974454|A|12.0.1.63|7018|191.242.88.0/22|7018 3356 28329 262612|IGP|12.0.1.63|0|0|7018:5000 7018:37232|NAG||',
    'BGP4MP|1569974454|A|12.0.1.63|7018|191.242.88.0/21|7018 3356 28329 262612|IGP|12.0.1.63|0|0|7018:5000 7018:38001|NAG||',
    'BGP4MP|1569974454|A|12.0.1.63|7018|99.194.200.0/22|7018 209 22561|IGP|12.0.1.63|0|0|7018:5000 7018:38001|NAG||',
    'BGP4MP|1569974454|A|12.0.1.63|7018|99.194.200.0/22|7018 209 22561|IGP|12.0.1.63|0|0|7018:5000 7018:37232|NAG||',
    'BGP4MP|1569974454|A|12.0.1.63|7018|84.205.67.0/24|7018 6830 50673 12654|IGP|12.0.1.63|0|0|7018:5000 7018:38001|NAG|64759 10.1.81.128|',
    'BGP4MP|1569974454|A|12.0.1.63|7018|84.205.69.0/24|7018 6453 47692 12654|IGP|12.0.1.63|0|0|7018:5000 7018:37232|NAG|64887 10.1.81.128|',
    'BGP4MP|1569974454|A|12.0.1.63|7018|93.175.149.0/24|7018 3257 12779 12654|IGP|12.0.1.63|0|0|7018:5000 7018:37232|NAG|64887 10.1.81.128|'
]
# define class labels
# labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
labels = array([1, 1, 1, 1, 0, 0, 0, 0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
