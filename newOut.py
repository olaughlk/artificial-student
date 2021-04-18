import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

artificial_student_model = tf.saved_model.load('artificial-student-model')

start = time.time()
states = None
next_char = tf.constant(['GVSU'])
result = [next_char]

for n in range(1,1000):
    next_char, states = artificial_student_model.generate_one_step(inputs=next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time: ', end - start)