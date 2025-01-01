from pyspark import SparkConf, SparkContext
import sys
import json

# Ensure Python 3.5+ is used
assert sys.version_info >= (3, 5)

# Parse JSON File and Return Score, Count, Author
def parse_json(line):
    try:
        record = json.loads(line)
        subreddit = record['subreddit']
        score = record['score']
        author = record['author']
        return (subreddit, (score, 1, author))
    except (json.JSONDecodeError, KeyError):
        return None

def parse_comments(line):
    try:
        record = json.loads(line)
        subreddit = record['subreddit']
        # Return subreddit as key and the entire record as value
        return (subreddit, record)  
    except (json.JSONDecodeError, KeyError):
        return None

def main(inputs, output):
    input_rdd = sc.textFile(inputs)
    subreddit_scores_rdd = input_rdd.map(parse_json).filter(lambda x: x is not None)

    # Reducer step: Summing scores and counts
    aggregated_rdd = subreddit_scores_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    # Compute the average score for each subreddit
    average_rdd = aggregated_rdd.mapValues(lambda x: x[0] / x[1])

    # Filter out subreddits with average scores <= 0
    filtered_average_rdd = average_rdd.filter(lambda x: x[1] > 0)

    comment_data_rdd = input_rdd.map(parse_comments).filter(lambda x: x is not None).cache()

    # Join the average scores with the comments
    joined_rdd = filtered_average_rdd.join(comment_data_rdd)

    # Calculate relative score and get the author's name
    relative_score_rdd = joined_rdd.map(lambda x: (
        x[1][1]['score'] / x[1][0],
        x[1][1]['author']
    ))

    # Sort by relative score in descending order
    sorted_rdd = relative_score_rdd.sortByKey(ascending=False)

    # Save the final result to the output path
    sorted_rdd.saveAsTextFile(output)

if __name__ == '__main__':
    conf = SparkConf().setAppName('relative_score')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    assert sc.version >= '3.0'  # Ensure Spark 3.0+ is used
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
