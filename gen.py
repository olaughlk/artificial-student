import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

path_to_file='./scraped_text.txt'

#just checking
#read text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#get all chars
vocab = sorted(set(text))

#process
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
chars

#setting up our converters
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)

ids = ids_from_chars(chars)


chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary = ids_from_chars.get_vocabulary(),
    invert = True
)

chars = chars_from_ids(ids)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis = -1)

#creating our ID dataset
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100

examples_per_epoch = len(text)//seq_length

sequences = ids_dataset.batch(seq_length+1, drop_remainder = True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#creating our batches

#Batch size
BATCH_SIZE = 64

#Buffer size to shuffle the dataset
#we need this so that TF isn't using the whole thing
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

#Building our model

#length of our vocabulary
vocab_size = len(vocab)

#embedding dimension
embedding_dim = 256

#number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences = True,
            return_state = True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state = False, training = False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training = training)

        if(return_state):
            return x, states
        else:
            return x

model = MyModel(
    vocab_size = len(ids_from_chars.get_vocabulary()),
    embedding_dim = embedding_dim,
    rnn_units = rnn_units
)

### GO HERE TO TEST IF NECESSARY ###

#model training

loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True)

#model compilation

model.compile(optimizer = 'adam', loss = loss)

#Checkpoints
#this is how we save our model while it is running
#without checkpoints, everything would be lost if it crashes

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True
)

EPOCHS = 1

history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])


model.summary()

###########

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__(self)
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['','[UNK]'])[:,None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states,return_state=True)

        predicted_logits = predicted_logits[:,-1, :]
        predicted_logits = predicted_logits/self.temperature

        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

tf.keras.models.save_model(one_step_model,'artificial-student-model', save_format="h5")

start = time.time()
states = None
next_char = tf.constant(['GVSU'])
result = [next_char]

for n in range(1,1000):
    next_char, states = one_step_model.generate_one_step(inputs=next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time: ', end - start)
