import sys
assert sys.version_info >= (3, 5)
import os
import re
import gzip
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from cassandra import ConsistencyLevel


line_re = re.compile(r'^(\S+) - - \[(\S+) [+-]\d+\] \"[A-Z]+ (\S+) HTTP/\d\.\d\" \d+ (\d+)$')
datetime_format = '%d/%b/%Y:%H:%M:%S'


def main(input_dir, keyspace, table):

    cluster = Cluster(['node1.local', 'node2.local'])
    session = cluster.connect(keyspace)

    # Prepare the insert statement
    insert_query = session.prepare(f"INSERT INTO {table} (id, host, datetime, path, bytes) VALUES (uuid(), ?, ?, ?, ?)")
    print(f"Looking for files in: {input_dir}")

    for f in os.listdir(input_dir):
        with gzip.open(os.path.join(input_dir, f), 'rt', encoding='utf-8') as logfile:
            batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)
            count = 0
            
            for line in logfile:
                match = line_re.match(line)
                if match:
                    host, datetime_str, path, bytes_transferred = match.groups()
                    datetime_obj = datetime.strptime(datetime_str, datetime_format)
                    
                    batch.add(insert_query, (host, datetime_obj, path, int(bytes_transferred)))
                    count += 1
                    
                    if count >= 200:  # Execute the batch every 200 entries
                        session.execute(batch)
                        batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)
                        count = 0
            if count > 0:
                session.execute(batch)

if __name__ == '__main__':
    input_dir = sys.argv[1]
    keyspace = sys.argv[2]
    table = sys.argv[3]
    main(input_dir, keyspace, table)