import praw
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

import sys

#######################################################################################
###                                                                                 ###
### Reddit scraper runs right before training begins ensuring content is up to date ###
###                                                                                 ###
#######################################################################################
f = open("scraped_text.txt", "w")

reddit = praw.Reddit(client_id="Ef1PrOTN25smOg",      # your client id
                     client_secret="twHqdYKqwp-OBBxr43Dk3lpg8as9Ww",  #your client secret
                     user_agent="my user agent", #user agent name
                     username = "Efficient_Natural302",     # your reddit username
                     password = "8qEsURjhjKLQJ7j")     # your reddit password


subreddit = reddit.subreddit('GVSU')


controversial_py = subreddit.controversial(limit=100)
top_py = subreddit.top(limit=100)


post_dict = {
    'body':[]
}

comments_dict = {
    'comment_body':[]
}

#######################################################################################
#                                                                                     #
# I wanted a variety of different opinions of students,  both supportive and critical #
# of the university. For this reason, I chose to include the top posts as well as the #
# most controversial posts. This adds a variety of content, and a more realistic look #
# at the opinions of the student body.                                                #
#                                                                                     #
#######################################################################################

for submission in controversial_py:
    post_dict["body"].append(submission.selftext)
    if submission.selftext != "" and submission.selftext[0].isalpha():
        f.write(submission.selftext)
        f.write("\n")

    ##### Acessing comments on the post
    submission.comments.replace_more(limit = 1)
    for comment in submission.comments.list():
        #comments_dict["comment_id"].append(comment.id)
        #comments_dict["comment_parent_id"].append(comment.parent_id)
        comments_dict["comment_body"].append(comment.body)
        #comments_dict["comment_link_id"].append(comment.link_id)
        if comment.body != "" and comment.body[0].isalpha():
            f.write(comment.body)
            f.write("\n")

for submission in top_py:
    post_dict["body"].append(submission.selftext)
    if submission.selftext != "" and submission.selftext[0].isalpha():
        f.write(submission.selftext)
        f.write("\n")

    ##### Acessing comments on the post
    submission.comments.replace_more(limit = 1)
    for comment in submission.comments.list():
        #comments_dict["comment_id"].append(comment.id)
        #comments_dict["comment_parent_id"].append(comment.parent_id)
        comments_dict["comment_body"].append(comment.body)
        #comments_dict["comment_link_id"].append(comment.link_id)
        if comment.body != "" and comment.body[0].isalpha():
            f.write(comment.body)
            f.write("\n")


post_comments = pd.DataFrame(comments_dict)

# I chose to write to an external file in case we want to reference it later

post_comments.to_csv('GVSU'+"_comments_subreddit.csv")
post_data = pd.DataFrame(post_dict)
post_data.to_csv('GVSU'+"_subreddit.csv")
f.close()

#######################################################################################
###                                                                                 ###
###        The Training model now runs, using the most recently saved data.         ###
###                                                                                 ###
#######################################################################################

path_to_file='./scraped_text.txt'
epoch_value = 50

#read text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#get all chars
vocab = sorted(set(text))

#process
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

#setting up our converters
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)

ids = ids_from_chars(chars)


chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary = ids_from_chars.get_vocabulary(),
    invert = True
)

#this is here to ensure it works, chars should not change value
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

########################
#                      #
#  Building our model  #
#                      #
########################

#length of our vocabulary
vocab_size = len(vocab)

#embedding dimension
embedding_dim = 256

#number of RNN units
rnn_units = 1024

#######################################################################################
#                                                                                     #
#   Here  we are building our  own custom  model.  This will  limit us  later on in   #
#   the process ( we'll get to that later ), but right now it is the easiest way to   #
#   ensure that we are training exactly the way we want to.  The model does inherit   #
#  functionality  from the base  tf.keras.Model class,  which is  what we will  use   #
#   when we actually call training algorithms.                                        #
#                                                                                     #
#######################################################################################

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


#model training

loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True)

#model compilation

model.compile(optimizer = 'adam', loss = loss)

#Checkpoints
#this is how we save our model while it is running
#without checkpoints, everything would be lost if it crashes

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#######################################################################################
#                                                                                     #
# Here is where we  run into  the  problem  with  using a custom  model. The standard #
# methods  for  saving models are through  either a hdf5 file format  or throufh a tf #
# format.  In order  to do that with  a  custom model,  we  need to cast it to one of #
# those. This model for some reason doesn't play well with either of those formats so #
# we have to  save weights only.  This just means that if we want to reload the model #
# we will have to redefine it, and load the weights in that way. Not a big issue, but #
# not ideal either.                                                                   #
#                                                                                     #
#######################################################################################


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True
)

EPOCHS = epoch_value

history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])

#This isn't necessary, but I like seeing the results of the model
model.summary()


########################
#                      #
#   Using our model    #
#                      #
########################

#######################################################################################
#                                                                                     #
# We define a new model here  that won't  actually  be used for training.  Instead it #
# will  be used to access abd utilize the already trained data.  Once again thie is a #
# custom model,  which makes exporting it difficult,  but for our purposes,  we don't #
# really need to do that anyways. The model has one function that we use to determine #
# the next character in the output.  This function takes in the current text contents #
# and outputs its prediction of what  should come  next.  We chose characters here as #
# opposed to words, for the purpose of minimizing the necessary dictionary. This does #
# mean that on occasion the model will output words that aren't real, but the benefit #
# of limiting memory, outweighs that.                                                 #
#                                                                                     #
#######################################################################################

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

# Here I added some starter words that the model can use to kickstart a sentence
states = None
next_chars = ['GVSU', 'University', 'I', 'We', 'Students', 'Grand', 'With', 'Professors', 'COVID', 'Pandemic',  'Together', 'My', 'Today', 'Class', 'When', 'Internship', 'Should', 'A', 'Can', 'Tuition']
next_char = tf.constant([next_chars[0]])
result = [next_char]

# We output to a file here as opposed to the terminal so to easily save our "tweets"
tweets = open("tweet_contents.txt", 'w')
ostd = sys.stdout
sys.stdout = tweets
for i in range(100):
    next_char = tf.constant([next_chars[i%20]])
    result = [next_char]
    for n in range(1,280):
        next_char, states = one_step_model.generate_one_step(inputs=next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print()
sys.stdout = ostd
tweets.close()

#################################################################################################
#                                                                                               #
# Referenced works:                                                                             #
#       https://medium.com/swlh/scraping-reddit-using-python-57e61e322486                       #
#       https://www.tensorflow.org/tutorials/text/text_generation                               #
#       https://towardsdatascience.com/natural-language-processing-with-tensorflow-e0a701ef5cef #
#                                                                                               #
#################################################################################################