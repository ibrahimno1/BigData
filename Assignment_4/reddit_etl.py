from pyspark import SparkConf, SparkContext
import sys
import json

# Ensure Python 3.5+ is used
assert sys.version_info >= (3, 5)

# Parse JSON and return a dictionary with 'subreddit', 'author', and 'score'
def split(text):
    subreddit = text.get("subreddit")
    score = text.get("score")
    author = text.get("author")
    return {"subreddit": subreddit, "score": score, "author": author}

# Parse JSON string into a dictionary
def parse_json(line):
    try:
        record = json.loads(line)
        return split(record)
    except (json.JSONDecodeError, KeyError):
        return None

# Function to check if a subreddit contains the letter 'e'
def contains_e(record):
    subreddit = record.get("subreddit", "").lower()
    return 'e' in subreddit

# Function to check if score is greater than 0
def is_positive(record):
    return record.get("score", 0) > 0

# Function to check if score is less than or equal to 0
def is_negative(record):
    return record.get("score", 0) <= 0

def main(inputs, output):
    # Load input data
    input_rdd = sc.textFile(inputs)

    # Parse JSON and filter out None values
    parsed_rdd = input_rdd.map(parse_json).filter(lambda x: x is not None)

    # Filter subreddits that contain 'e'
    filtered_rdd = parsed_rdd.filter(contains_e).cache()

    # Separate data into positive and negative scores
    positive_rdd = filtered_rdd.filter(is_positive)
    negative_rdd = filtered_rdd.filter(is_negative)

    # Write positive rows to the 'positive' directory in JSON format
    positive_rdd.map(json.dumps).saveAsTextFile(output + '/positive')

    # Write negative rows to the 'negative' directory in JSON format
    negative_rdd.map(json.dumps).saveAsTextFile(output + '/negative')


if __name__ == '__main__':
    conf = SparkConf().setAppName('reddit_etl')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    assert sc.version >= '3.0'  # Ensure Spark 3.0+ is used
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)