import sys
from pyspark.sql import SparkSession, functions as F, types as T

def source_dest(line):
    if ':' not in line:
        return
    pairs = line.split(':')
    if len(pairs) < 2:
        return
    source = pairs[0]
    dest = pairs[1].split(" ")
    for value in dest:
        if value != "":
            try:
                yield (int(source), int(value))
            except ValueError:
                continue


def load_graph_edges(spark, inputs):
    lines = sc.textFile(inputs)
    node_dest_pair = lines.flatMap(source_dest) 
    edges_schema = T.StructType([
        T.StructField('node', T.IntegerType(), False),
        T.StructField('neighbor', T.IntegerType(), False)
    ])
    edges_df = spark.createDataFrame(node_dest_pair, schema=edges_schema).cache()
    #edges_df.show()
    return edges_df


def initialize_paths_df(spark, source):
    #print(f"Source value: {source}")
    #print(f"Type of source: {type(source)}")
    paths_schema = T.StructType([
        T.StructField('node', T.IntegerType(), False),
        T.StructField('source', T.IntegerType(), False),
        T.StructField('distance', T.IntegerType(), False)
    ])
    #print(f"Creating DataFrame with: ({source}, {source}, 0)")
    paths_df = spark.createDataFrame([(source, source, 0)], schema=paths_schema).cache()
    return paths_df

def find_shortest_paths(spark, paths_df, edges_df, output, destination):
    path_found = False
    for i in range(6):

        paths_alias = paths_df.alias('paths')
        edges_alias = edges_df.alias('edges')
        

        new_paths_df = paths_alias.join(
            edges_alias, 
            paths_alias['node'] == edges_alias['node'], 
            'inner'
        ).select(
            F.col('edges.neighbor').alias('node'),
            F.col('paths.node').alias('source'),
            (F.col('paths.distance') + 1).alias('distance')
        )

        paths_df = paths_df.unionAll(new_paths_df)
        #paths_df.show()
        paths_df = paths_df.groupBy('node').agg(
            F.first('source').alias('source'),
            F.min('distance').alias('distance')
        )


        paths_df.write.csv(f'{output}/iter-{i}', mode='overwrite')

        if paths_df.filter(F.col('node') == destination).count() > 0:
            path_found = True
            break
        #paths_df.show()
            
    return paths_df, path_found


def main(inputs, output, source, destination):
    edges_df = load_graph_edges(spark, inputs)
    paths_df = initialize_paths_df(spark, source)
    paths_df,path_found = find_shortest_paths(spark, paths_df, edges_df, output, destination)

    if not path_found:
        print("No path found")
        return

    final_path = []
    current_node = destination
    while current_node != source:
        row = paths_df.filter(paths_df['node'] == current_node).collect()[0]
        final_path.append(current_node)
        current_node = row['source']

    final_path.append(source)
    final_path.reverse()

    final_path_rdd = sc.parallelize(final_path)
    final_path_rdd.saveAsTextFile(f'{output}/path')

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    source = int(sys.argv[3])
    destination = int(sys.argv[4])
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0'
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output, source, destination)

