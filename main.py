### THESE ARE ONLY SUGGESTED IMPORTS ###

# web (db and server) imports
from flask import Flask, render_template, request, url_for, jsonify, make_response
import pymysql
from pymongo import MongoClient
import urllib
# machine learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import time
# helpers
from collections import Counter
from datetime import datetime
# need pickle to store (if you want) binary files in mongo
import pickle
# json/bson handling
import json
from bson import ObjectId
from bson.decimal128 import Decimal128
from math import floor

import credentials

### HELPER FUNCTIONS ###

# you need this to decode mongo objs to JSON so they can render in the browser
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

# function to get the current time as a string
def get_current_time():
	# datetime object containing current date and time
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	return dt_string

username=credentials.USERNAME
password=credentials.MYSQL_PASSWORD
mongo_db_password=credentials.MONGO_PASSWORD
mongo_db_host=credentials.MONGO_DB_HOST
# Connect to the database
connection = pymysql.connect(host=credentials.HOST_NAME,
                             user=username,
                             password=password,
                             db=credentials.DB_NAME,
                             charset='utf8mb4'
                             )

### MONGODB CONNECTION ###
client = MongoClient(f'mongodb://{username}:{mongo_db_password}@{mongo_db_host}:27017/{username}?auth_source={username}', ssl=True)

print('Connected to the MongoDB database :)')
# create an object of our database
mdb = client.c2075016

category_map={0:"not covid", 1:"covid"}

app = Flask(__name__)

# Adapted from: https://stackoverflow.com/questions/14853694/python-jsonify-dictionary-in-utf-8
# This allows emojis and special characters to be rendered in the browser when a JSON is returned.
app.config['JSON_AS_ASCII'] = False

def reset_views():
    # your code here
    print("Dropping views...")
    with connection.cursor() as cur:
        q="""DROP VIEW IF EXISTS training_set;"""
        cur.execute(q)

        q="""DROP VIEW IF EXISTS test_set;"""
        cur.execute(q)

        connection.commit()


    print('Successfully dropped training_data and test_data views')

def create_labelling_function():

    print("Creating labelling function...")

    with connection.cursor() as cur:
        q="""DROP FUNCTION IF EXISTS assign_label;"""
        cur.execute(q)

        q= """CREATE FUNCTION assign_label(ID_of_post INT)
            RETURNS INT
            DETERMINISTIC
            BEGIN
                DECLARE label INT;
                SET @subreddit_name = (SELECT subr_name FROM posts INNER JOIN subreddits ON subreddits.subr_ID=posts.subreddit_ID WHERE post_ID=ID_of_post);
                SET @subreddit_description = (SELECT subr_description FROM posts INNER JOIN subreddits ON subreddits.subr_ID=posts.subreddit_ID WHERE post_ID=ID_of_post);

                IF @subreddit_name LIKE '%covid%' OR @subreddit_name LIKE '%corona%' OR @subreddit_name LIKE '%lockdown%' OR @subreddit_description LIKE '%covid%' OR @subreddit_description LIKE '%corona%' OR @subreddit_description LIKE '%lockdown%'
                THEN SET label=1;
                ELSE
                    SET label=0;
                END IF;
                RETURN label;
            END;"""
        cur.execute(q)

    print("Successfully created labelling function")

def create_training_view():
	# your code here

    print("Creating training view...")

    with connection.cursor() as cur:
        q="""SELECT COUNT(*) FROM posts;"""
        cur.execute(q)
        total_posts=cur.fetchone()[0]
        print(total_posts)

        q="""SELECT COUNT(*) FROM posts WHERE assign_label(post_ID)=1;"""
        cur.execute(q)
        total_covid_posts=cur.fetchone()[0]

        total_non_covid_posts=total_posts-total_covid_posts

        train_test_proportion=4/5
        train_covid_posts=floor(train_test_proportion*total_covid_posts)
        test_covid_posts=total_covid_posts-train_covid_posts
        train_non_covid_posts=floor(train_test_proportion*total_non_covid_posts)
        test_non_covid_posts=total_non_covid_posts-train_non_covid_posts

        q="""CREATE VIEW training_set AS
                (SELECT CONCAT_WS('\n', title, selftext) AS full_text, assign_label(post_ID) AS label FROM posts
                    WHERE assign_label(post_ID)=1
                    ORDER BY posted_at ASC, full_text ASC
                    LIMIT {0})
                UNION ALL
                (SELECT CONCAT_WS('\n', title, selftext) AS full_text, assign_label(post_ID) AS label FROM posts
                    WHERE assign_label(post_ID)=0
                    ORDER BY posted_at ASC, full_text ASC
                    LIMIT {1});
        """
        print(q.format(train_covid_posts, train_non_covid_posts))
        cur.execute(q.format(train_covid_posts, train_non_covid_posts))
        connection.commit()

    print('Successfully created training set')


def create_test_view():
	# your code here
    print("Creating test view...")

    with connection.cursor() as cur:
        q="""SELECT COUNT(*) FROM posts;"""
        cur.execute(q)
        total_posts=cur.fetchone()[0]

        q="""SELECT COUNT(*) FROM posts WHERE assign_label(post_ID)=1;"""
        cur.execute(q)
        total_covid_posts=cur.fetchone()[0]

        total_non_covid_posts=total_posts-total_covid_posts

        train_test_proportion=4/5
        train_covid_posts=floor(train_test_proportion*total_covid_posts)
        test_covid_posts=total_covid_posts-train_covid_posts
        train_non_covid_posts=floor(train_test_proportion*total_non_covid_posts)
        test_non_covid_posts=total_non_covid_posts-train_non_covid_posts

        q="""CREATE VIEW test_set AS
            (SELECT CONCAT_WS('\n', title, selftext) AS full_text, assign_label(post_ID) AS label FROM posts
                WHERE assign_label(post_ID)=1
                ORDER BY posted_at DESC, full_text DESC
                LIMIT {0})
            UNION ALL
            (SELECT CONCAT_WS('\n', title, selftext) AS full_text, assign_label(post_ID) AS label FROM posts
                WHERE assign_label(post_ID)=0
                ORDER BY posted_at DESC, full_text DESC
                LIMIT {1});
        """
        print(q.format(test_covid_posts, test_non_covid_posts))
        cur.execute(q.format(test_covid_posts, test_non_covid_posts))
        connection.commit()

    print('Successfully created test set')

def check_views():
	# your code here
    print("Checking views...")

    with connection.cursor() as cur:
        q="""SELECT COUNT(*) FROM training_set;"""
        cur.execute(q)
        training_posts=cur.fetchone()[0]

        q="""SELECT COUNT(*) FROM test_set;"""
        cur.execute(q)
        test_posts=cur.fetchone()[0]

    print('Training data size: ',training_posts)
    # your code here
    print('Test data size: ',test_posts)


@app.route('/')
def form():
    reset_views()
    # Added create_labelling_function in to create the labelling function.
    create_labelling_function()

    create_training_view()
    create_test_view()
    check_views()
    return render_template('index.html')

# Trains a classifier using the parameters input by the user, which is saved to the MongoDB database.
@app.route('/experiment_done', methods=['POST'])
def experiment_done():
	# your code here
    print("Retrieving data...")
    with connection.cursor() as cur:
        q="""SELECT * FROM training_set;"""
        cur.execute(q)
        train_data=cur.fetchall()

        q="""SELECT * FROM test_set;"""
        cur.execute(q)
        test_data=cur.fetchall()

    train_full_text=[row[0] for row in train_data]
    train_labels=[row[1] for row in train_data]
    test_full_text=[row[0] for row in test_data]
    test_labels=[row[1] for row in test_data]


    print("Choosing features and vectorizing data...")
    vectorizer=TfidfVectorizer(max_features=1000, stop_words='english')
    X=vectorizer.fit_transform(train_full_text)
    X_test=vectorizer.transform(test_full_text)

    print("Training classifier...")

    start=time.time()
    max_depth=int(request.form["max_depth"])
    n_estimators=int(request.form["n_estimators"])
    max_features=int(request.form["max_features"])

    clf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
    clf=clf.fit(X, train_labels)

    print("Evaluating classifier...")
    predicted_labels=clf.predict(X_test)
    accuracy=accuracy_score(test_labels, predicted_labels)
    precision=precision_score(test_labels, predicted_labels, average='macro')
    recall=recall_score(test_labels, predicted_labels, average='macro')
    f1score=f1_score(test_labels, predicted_labels, average='macro')
    end=time.time()
    print(end-start, accuracy, precision, recall, f1score)

    print("Saving model...")
    model_binary=pickle.dumps(clf)
    vectorizer_binary=pickle.dumps(vectorizer)

    record_to_store={"model": {"binary": model_binary, "classifier": "RandomForestClassifier", "parameters": {"max_depth":max_depth, "n_estimators":n_estimators, "max_features":max_features}}, "vectorizer": {"binary": vectorizer_binary, "type": "TfidfVectorizer", "parameters": {"max_features":1000, "stop_words":"english"}}, "train_test_proportion":4/5, "evaluation_metrics": {"time":end-start, "accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}}

    record_to_show={"model": {"classifier": "RandomForestClassifier", "parameters": {"max_depth":max_depth, "n_estimators":n_estimators, "max_features":max_features}}, "vectorizer": {"type": "TfidfVectorizer", "parameters": {"max_features":1000, "stop_words":"english"}}, "train_test_proportion":4/5, "evaluation_metrics": {"time":end-start, "accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}}

	# you will be saving binary files in your mongodb database
	# it is ok to have two different records, one with the models and one without it, so you store the one WITH models,
	# and show in the browser the one WITHOUT models
    mdb.results.insert_one(record_to_store)
    return JSONEncoder().encode(record_to_show)

# Retrieves the top 3 models from MongoDB with the highest F1-scores and they are rendered on the page.
@app.route('/report', methods=['GET', 'POST'])
def retrieve_results():
# 	# your code here
    projection={"_id":0, "model.classifier":1, "model.parameters":1, "vectorizer.type":1, "vectorizer.parameters":1, "train_test_proportion":1, "evaluation_metrics":1}
    res=list(mdb.results.find({}, projection).sort("evaluation_metrics.f1score", -1).limit(3))
    return JSONEncoder().encode(res)

# Retrieves the model and vectorizer binaries for the best model in terms of F1-score compiles them using pickle.loads
# and generates a prediction for the value of "input_text" submitted in the form, which is then rendered in the browser
# along with "input_text".
@app.route('/submitted', methods=['POST'])
def submitted_form():
# 	# your code here
    projection={"_id":0, "model.binary":1, "vectorizer.binary":1}
    result=list(mdb.results.find({}, projection).sort("evaluation_metrics.f1score", -1).limit(1))[0]

    vectorizer_binary=result["vectorizer"]["binary"]
    vectorizer=pickle.loads(vectorizer_binary)

    best_clf_binary=result["model"]["binary"]
    best_clf=pickle.loads(best_clf_binary)

    post=request.form["input_text"]

    X=vectorizer.transform([post])
    prediction_key=best_clf.predict(X)[0]
    print(prediction_key)
    prediction=category_map[prediction_key]
    res={"input_text": post, "prediction": prediction}
    return jsonify(res)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
