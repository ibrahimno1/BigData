from pyspark import SparkConf, SparkContext
import sys

# Read the input file(s) in as lines (as in the word count).
inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('wikipedia_popular')
sc = SparkContext(conf=conf)
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.3'  # make sure we have Spark 2.3+

# Break each line up into a tuple of five things (by splitting around spaces). This would be a good time to convert he view count to an integer. (.map())
def split_line(line):
    wiki = line.split()  # Split the line by spaces
    if len(wiki) == 5:
        time = wiki[0]
        lang = wiki[1].lower()
        titled = wiki[2]
        requested = int(wiki[3])  # Convert to integer
        size = wiki[4]
        return (time, lang, titled, requested, size)  # Return a tuple with the processed elements
    else:
        return None  # Handle lines that don't have exactly five elements

# (.map()) and Remove the records we don't want to consider. (.filter())
def process_valid_lines(rdd):
    return rdd.map(split_line) \
              .filter(lambda x: x is not None and x[1] == 'en' and x[2] != 'Main_Page' and not x[2].startswith('Special:')) \
              .map(lambda x: (x[0], (x[3], x[2])))  # Create key-value pairs

#Create an RDD of key-value pairs. (.map())
# Read the input file into an RDD
lines_rdd = sc.textFile(inputs)
key_value_rdd = process_valid_lines(lines_rdd)

# Reduce to find the max value for each key. (.reduceByKey())
max_requested_rdd = key_value_rdd.reduceByKey(lambda a, b: a if a[0] > b[0] else b)

# Sort so the records are sorted by key. (.sortBy())
sorted_rdd = max_requested_rdd.sortBy(lambda x: x[0])

# Save as text output
def tab_separated(kv):
    return "%s\t(%s, '%s')" % (kv[0], kv[1][0], kv[1][1])

sorted_rdd.map(tab_separated).saveAsTextFile(output)