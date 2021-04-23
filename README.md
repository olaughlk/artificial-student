# **Artificial Student TensorFlow NLP Generator**

### **Purpose**

This project was created as a test of how accurately a TensorFlow Keras model could emulate the language and demeanor of a student at Grand Valley State University. As my honors senior project, I felt it would be interesting and informative to identify trends in the GVSU online community.

---

### **Methods**
Gathering data from the [r/GVSU](https://reddit.com/r/gvsu "Reddit.com | Grand Valley State University") subreddit using the [PRAW](https://praw.readthedocs.io/en/latest/ "PRAW: The Python Reddit API Wrapper") Reddit wrapper. I selected a variety of content, including popular and unpopular posts on the sub. The reason for this was to get as accurate an understanding of community thought and expressions.

Data was then processed through the [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras "Module: tf.keras | TensorFlow Core v2.4.1") modeling libraries. These libraries were used to make custom models for Natural Language Processing and text generation. The trained data was sent through an additional model that then generated new text representing the processed data.

The Model outputs data into a text file containing 100 tweet-sized paragraphs. These can be tweeted while the script is run again to generate more. Tweets will be saved unitl the script is run again, so there will always be available data.
###### *data will be overwritten once script has been run.

---

### **Code**

This project is written entirely in Python, and can be consolidated down into one script. The script will require you to have several libraries installed for it to run.

#### **For Reddit scraper**

[pandas](https://pandas.pydata.org/ "pandas - Python Data Analysis Library")

[PRAW](https://praw.readthedocs.io/en/latest/ "PRAW: The Python Reddit API Wrapper")

#### **For NLP Model**
[TensorFlow](https://www.tensorflow.org/ "TensorFlow")

[TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras "Module: tf.keras | TensorFlow Core v2.4.1")

[NumPy](https://numpy.org/ "NumPy")

#### **Execution Requirements**
Reddit account and developer API key are required, they will need to be filled in in the following block of code:
```Python
reddit = praw.Reddit(
    client_id="",      # your client id
    client_secret="",  #your client secret
    user_agent="", #user agent name
    username = "",     # your reddit username
    password = ""  # your reddit password
)    
```
All of this information is easily accessable when you sign up for a developer account [here](https://www.reddit.com/wiki/api "api - reddit.com").

It should be noted that the model is currently set to run for 50 Epochs, or it will run through the training algorithm 50 times. While I have found this to be the most consistant length, results will vary each time it is run. For example, one run may produce results akin to:

    I’ve talked to someone today and I just felt so sad and angry.

The same model may generate:

    They gad a cas has been an increased chill, cool day

Results will not be entirely consistent, which is part of the reason we generate 100 of them.

---

I hope you find this little experiment to be just as fun as I did, and hopefully learn something along the way. References used when researching and building this can be found at:

https://medium.com/swlhscraping-reddit-using-python-57e61e322486                       
https://www.tensorflow.org/tutorials/text/text_generation                               
https://towardsdatascience.com/natural-language-processing-with-tensorflow-e0a701ef5cef