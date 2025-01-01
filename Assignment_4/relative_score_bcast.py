from pyspark import SparkConf, SparkContext
import sys
import json

# Ensure Python 3.5+ is used
assert sys.version_info >= (3, 5)

# Return (score, count, author)
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
        return (subreddit, record)  
    except (json.JSONDecodeError, KeyError):
        return None

def main(inputs, output):
    input_rdd = sc.textFile(inputs)
    subreddit_scores_rdd = input_rdd.map(parse_json).filter(lambda x: x is not None)

    # Reducer step: summing scores and counts
    aggregated_rdd = subreddit_scores_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1]) 
    )

    # Compute the average score for each subreddit
    average_rdd = aggregated_rdd.mapValues(lambda x: x[0] / x[1])

    # Filter out subreddits with average scores <= 0
    filtered_average_rdd = average_rdd.filter(lambda x: x[1] > 0)

    # filtered_average_rdd is small - we are allowed to use collect()
    averages_dict = dict(filtered_average_rdd.collect())
    averages_broadcast = sc.broadcast(averages_dict)

    comment_data_rdd = input_rdd.map(parse_comments).filter(lambda x: x is not None).cache()

    # Join the comment data RDD with the broadcasted averages
    def calculate_relative_score(subreddit, record):
        avg_score = averages_broadcast.value.get(subreddit, None)
        if avg_score is not None:
            comment_score = record['score']
            author = record['author']
            relative_score = comment_score / avg_score
            return (relative_score, author)
        else:
            return None

    # Apply the function to each comment record
    relative_score_rdd = comment_data_rdd.map(lambda x: calculate_relative_score(x[0], x[1])).filter(lambda x: x is not None)

    # Sort by relative score in descending order
    sorted_rdd = relative_score_rdd.sortByKey(ascending=False)

    # Save the final result to the output path
    formatted_output = sorted_rdd.map(lambda x: f'{x[0]}\t{x[1]}')
    formatted_output.saveAsTextFile(output)

if __name__ == '__main__':
    conf = SparkConf().setAppName('relative_score_bcast')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    assert sc.version >= '3.0'  # Ensure Spark 3.0+ is used
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
