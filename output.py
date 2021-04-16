
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

path_to_file='./scraped_text.txt'

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
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary = ids_from_chars.get_vocabulary(),
    invert = True
)

model = keras.models.load_model('model_weights_saved.hdf5')

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
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

        predicted_logits, states, other = self.model(inputs=input_ids, states=states,return_state=True)

        predicted_logits = predicted_logits[:,-1, 1]
        predicted_logits = predicted_logits/self.temperature

        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['GVSU'])
result = [next_char]

for n in range(1,1000):
    #next_char, states, other = one_step_model.generate_one_step(inputs=next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy.decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time: ', end - start)