import praw
import pandas as pd


#https://medium.com/swlh/scraping-reddit-using-python-57e61e322486

f = open("scraped_text.txt", "a")

reddit = praw.Reddit(client_id="Ef1PrOTN25smOg",      # your client id
                     client_secret="twHqdYKqwp-OBBxr43Dk3lpg8as9Ww",  #your client secret
                     user_agent="my user agent", #user agent name
                     username = "Efficient_Natural302",     # your reddit username
                     password = "8qEsURjhjKLQJ7j")     # your reddit password


subreddit = reddit.subreddit('GVSU')


controversial_py = subreddit.controversial(limit=100)
top_py = subreddit.top(limit=100)


post_dict = {
    #'title':[],
    #'score':[],
    #'id':[],
    #'url':[],
    #'comms_num':[],
    #'created':[],
    'body':[]
}

comments_dict = {
    #'comment_id':[],
    #'comment_parent_id':[],
    'comment_body':[]
    #'comment_link_id':[]
}

#for submission in subreddit.search(query,sort = 'controversial',limit = 10):
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

post_comments.to_csv('GVSU'+"_comments_"+ item +"_subreddit.csv")
post_data = pd.DataFrame(post_dict)
post_data.to_csv('GVSU'+"_"+ item +"_subreddit.csv")
f.close()

