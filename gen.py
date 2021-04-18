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

model.save(filepath='./model_weights_saved.hdf5', save_format="tf")
