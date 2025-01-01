import java.io.IOException;
import javax.naming.Context;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class WikipediaPopular extends Configured implements Tool {

    public static class WikipediaMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private final static LongWritable viewCount = new LongWritable();
        private Text timestampKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Split the line with space
            String[] fields = value.toString().split(" ");
            if (fields.length < 5) {
                return; // Skip lines that do not have exactly 5 values
            }

            // Extract relevant fields
            String timestamp = fields[0];    // timestamp
            String lang = fields[1].toLowerCase();  // language code (converted to lowercase)
            String title = fields[2];        // page title
            long views = Long.parseLong(fields[3]);

            // Apply filters:Language must be "en" (in lowercase),Title must not be "Main_Page",Title must not start with "Special:"
            if (!"en".equals(lang) || "Main_Page".equals(title) || title.startsWith("Special:")) {
                return; // Skip this record if it doesn't meet the criteria
            }

            // Set the timestamp as key and view count as value
            timestampKey.set(timestamp);
            viewCount.set(views);

            // Contect write timestamp as key and view count as value
            context.write(timestampKey, viewCount);
        }
    }


    public static class WikipediaReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private LongWritable result = new LongWritable();

        @Override
        protected void reduce(Text key, Iterable<LongWritable> values, Context context) 
                throws IOException, InterruptedException {
            long maxViews = 0;

            // Find the max view count for each hour in the input file
            for (LongWritable val : values) {
                if (val.get() > maxViews) {
                    maxViews = val.get();
                }
            }

            result.set(maxViews);
            context.write(key, result); // Write the hour and max views
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: WikipediaPopular <input path> <output path>");
            return -1;
        }

        Job job = Job.getInstance(getConf(), "Wikipedia Popular");
        job.setJarByClass(WikipediaPopular.class);

        // Set mapper and reducer
        job.setMapperClass(WikipediaMapper.class);
        job.setReducerClass(WikipediaReducer.class);

        // Set output key and value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);

        // Set input and output format classes
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new WikipediaPopular(), args);
        System.exit(exitCode);
    }
}
