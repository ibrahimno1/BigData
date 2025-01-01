import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.json.JSONObject;

public class RedditAverage extends Configured implements Tool {

 
    public static class LongPairWritable implements Writable {
        private long scoreSum;
        private long count;

        public LongPairWritable(long scoreSum, long count) {
            this.scoreSum = scoreSum;
            this.count = count;
        }

        public LongPairWritable() {
            this(0, 0);
        }

        public long getScoreSum() {
            return scoreSum;
        }

        public long getCount() {
            return count;
        }

        public void set(long scoreSum, long count) {
            this.scoreSum = scoreSum;
            this.count = count;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeLong(scoreSum);
            out.writeLong(count);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            scoreSum = in.readLong();
            count = in.readLong();
        }

        @Override
        public String toString() {
            return "(" + scoreSum + "," + count + ")";
        }
    }

    // Mapper Class
    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, LongPairWritable> {

        private final static LongPairWritable longPairWritable = new LongPairWritable();
        private Text subredditKey = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();

            try {              
                JSONObject record = new JSONObject(line);
                String subreddit = record.getString("subreddit");
                int score = record.getInt("score");               
                subredditKey.set(subreddit);               
                longPairWritable.set(score, 1);               
                context.write(subredditKey, longPairWritable);
            } catch (Exception e) {
                System.err.println("Error parsing JSON: " + e.getMessage());
            }
        }
    }

    // Combiner Class
    public static class LongPairCombiner extends Reducer<Text, LongPairWritable, Text, LongPairWritable> {
        private LongPairWritable result = new LongPairWritable();
        @Override
        public void reduce(Text key, Iterable<LongPairWritable> values, Context context)
                throws IOException, InterruptedException {
            long sumScores = 0;
            long totalCount = 0;
            for (LongPairWritable value : values) {
                sumScores += value.getScoreSum();
                totalCount += value.getCount();
            }
            result.set(sumScores, totalCount);
            context.write(key, result);
        }
    }

    // Reducer Class
    public static class AverageReducer extends Reducer<Text, LongPairWritable, Text, DoubleWritable> {

        private DoubleWritable result = new DoubleWritable();

        @Override
        public void reduce(Text key, Iterable<LongPairWritable> values, Context context)
                throws IOException, InterruptedException {

            long sumScores = 0;
            long totalCount = 0;
            for (LongPairWritable value : values) {
                sumScores += value.getScoreSum();
                totalCount += value.getCount();
            }
            double averageScore = (double) sumScores / totalCount;
            result.set(averageScore);
            context.write(new Text(key.toString().toLowerCase()), result);
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: RedditAverage <input path> <output path>");
            return -1;
        }

        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Reddit Average");

        job.setJarByClass(RedditAverage.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(LongPairCombiner.class);
        job.setReducerClass(AverageReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(LongPairWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        TextInputFormat.addInputPath(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new RedditAverage(), args);
        System.exit(exitCode);
    }
}

