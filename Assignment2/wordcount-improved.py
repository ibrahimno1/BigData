from pyspark import SparkConf, SparkContext
import sys
import re, string

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('word count immproved')
sc = SparkContext(conf=conf)
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.3'  # make sure we have Spark 2.3+
wordsep = re.compile(r'[%s\s]+' % re.escape(string.punctuation))

def words_once(line):
    for w in wordsep.split(line):
        yield (w.lower(), 1)

def add(x, y):
    return x + y

def get_key(kv):
    return kv[0]

def output_format(kv):
    k, v = kv
    return '%s %i' % (k, v)

text = sc.textFile(inputs)
words = text.flatMap(words_once)
wordcount = words.reduceByKey(add)

text = sc.textFile(inputs)
words = text.flatMap(words_once)

# Remove empty strings and words consisting only of punctuation, but keep numbers
filtered_words = words.filter(lambda w: w[0] and not all(char in string.punctuation for char in w[0]))

# Perform the word count using the filtered words
wordcount = filtered_words.reduceByKey(add)

outdata = wordcount.sortBy(get_key).map(output_format)
outdata.saveAsTextFile(output)