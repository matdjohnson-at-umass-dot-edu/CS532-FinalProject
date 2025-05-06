import ir_datasets
from pyspark import SparkContext, StorageLevel
from pyspark import SparkConf
from pyspark.sql import SparkSession
from collections import Counter
import numpy as np

import mysql.connector

import re
import csv
import os
import time
import pickle
import math
import base64

# variables for non-experimental, trial execution (replication of code execution)
spark_host = "spark://10.0.0.166:7077"  # ignored for remote execution
max_cores = 12
max_mem_mb = 15000

# variables for conditional program flow
# remote execution is colab execution
# experiment execution is for execution of program with varying resource allocations (requires v-28 instance on colab, manual configuration otherwise)
remote_execution = False
run_multiple_configurations = True
memory_factor = 1.0
core_segments = min(max_cores, 12)
use_large_dataset = False
repartition_rdds = True
reparition_multiple_of_executors = 6
run_bm25_evaluation = False

# create directories for datasets and outputs
if remote_execution:
    root_directory = "/content/drive/MyDrive/CS532-FinalProject"
else:
    root_directory = "."
    mysql_connection_url = "jdbc:mysql://localhost:3306/wikipedia_docs"
dataset_directory = f"{root_directory}/data/ir_datasets"
log_directory = f"{root_directory}/output/macbook_standard_partitions"
if use_large_dataset:
    training_dataset_name = "wikir/en78k/training"
else:
    training_dataset_name = "wikir/en1k/training"
training_dataset_filename = training_dataset_name.replace("/", "_")
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# load dataset if dataset has not been loaded previously
if not (os.path.isfile(f"{dataset_directory}/{training_dataset_filename}_docs")
        and os.path.isfile(f"{dataset_directory}/{training_dataset_filename}_queries")
        and os.path.isfile(f"{dataset_directory}/{training_dataset_filename}_qrels")):
    train_dataset = ir_datasets.load(training_dataset_name)

    docs_file = open(f"{dataset_directory}/{training_dataset_filename}_docs", 'a')
    csv_writer = csv.writer(docs_file, dialect='unix')
    for doc in train_dataset.docs_iter():
        csv_writer.writerow([doc.doc_id, doc.text])
    docs_file.close()

    queries_file = open(f"{dataset_directory}/{training_dataset_filename}_queries", 'a')
    csv_writer = csv.writer(queries_file, dialect='unix')
    for query in train_dataset.queries_iter():
        csv_writer.writerow([query.query_id, query.text])
    queries_file.close()

    qrels_file = open(f"{dataset_directory}/{training_dataset_filename}_qrels", 'a')
    csv_writer = csv.writer(qrels_file, dialect='unix')
    for qrel in train_dataset.qrels_iter():
        csv_writer.writerow([qrel.query_id, qrel.doc_id, qrel.relevance, qrel.iteration])
    qrels_file.close()


# define Spark configurations
class SparkConfHolder:
    def __init__(self, cores_max, executor_cores, executor_instances, executor_memory, executor_pyspark_memory):
        self.cores_max = cores_max
        self.executor_cores = executor_cores
        self.executor_instances = executor_instances
        self.executor_memory = executor_memory
        self.executor_pyspark_memory = executor_pyspark_memory

    def get_conf(self):
        spark_conf = SparkConf()
        spark_conf.setAll([("spark.cores.max", f"{int(self.cores_max)}"),
                           ("spark.executor.cores", "1"),
                           ("spark.executor.instances", f"{int(self.executor_instances)}"),
                           ("spark.executor.memory", f"{int(self.executor_memory)}m"),
                           ("spark.executor.pyspark.memory", f"{int(self.executor_pyspark_memory)}m"),
                           ("spark.jars.packages", "mysql:mysql-connector-java:8.0.33")])
        return spark_conf

    def get_executor_count(self):
        return self.executor_instances

    def __repr__(self):
        return f"{int(self.cores_max)}cm-{int(self.executor_cores)}ec-{int(self.executor_instances)}ei-{int(self.executor_memory)}em-{int(self.executor_pyspark_memory)}epm"


spark_conf_holders = list()
executor_memory = int((math.floor(max_mem_mb/max_cores) * memory_factor // 100) * 100)
executor_pyspark_memory = int((((2/3) * math.floor(max_mem_mb/max_cores) * memory_factor) // 100) * 100)
if run_multiple_configurations:
    segments = min(max_cores, core_segments)
    core_diff_per_segment = max_cores // segments
    spark_conf_holders.append(SparkConfHolder(max_cores, 1, max_cores, executor_memory, executor_pyspark_memory))
    for i in range(segments - 1, 0, -1):
        spark_conf_holders.append(SparkConfHolder(i * core_diff_per_segment, 1, i * core_diff_per_segment, executor_memory, executor_pyspark_memory))
else:
    spark_conf_holders.append(SparkConfHolder(max_cores, 1, max_cores, executor_memory, executor_pyspark_memory))

# define methods for execution by RDDs

# map corpus documents to corpus vocabulary term and term posting pairs
# D -> (D.text.word, (D.doc_id, count(D.text.word | D.doc_id))
def inverted_index_map_function(csv_file_line):
    csv_file_line_elements = csv_file_line.split('\",\"')
    doc_id = re.sub("[^A-Za-z0-9 ]", "", csv_file_line_elements[0])
    words_counter_for_doc = Counter(re.sub("[^A-Za-z0-9 ]", "", csv_file_line_elements[1]).lower().split(' '))
    words_for_doc = list(words_counter_for_doc.keys())
    word_postings_for_doc = list([[doc_id, str(words_counter_for_doc[words_for_doc[i]])]] for i in range(0, len(words_for_doc)))
    return list(zip(words_for_doc, word_postings_for_doc))

# reduce corpus vocabulary term and corpus document id pairs to map of vocab terms to doc id lists
# list((term, (doc_id, term_count))) -> dict({term: list((doc_id, term_count))})
def inverted_index_reduce_function(list_of_doc_ids_for_term_instance_1, list_of_doc_ids_for_term_instance_2):
    list_of_doc_ids_for_term_instance_1 += list_of_doc_ids_for_term_instance_2
    return list_of_doc_ids_for_term_instance_1

# translate inverted index to MySQL storage format
def inverted_index_pickle_function(rdd_element):
    return (rdd_element[0], "{\"postings\": \"" + base64.b64encode(pickle.dumps(rdd_element[1])).decode('utf-8') + "\"}")

# return RDD with document lengths
# used for computing the max and mean document lengths for use in the BM25 algorithm
def compute_doc_lengths(csv_file_line):
    csv_file_line_elements = csv_file_line.split('\",\"')
    words_for_doc = list(set(re.sub("[^A-Za-z0-9 ]", "", csv_file_line_elements[1]).lower().split(' ')))
    return len(words_for_doc)

# return RDD with mappings of doc_ids on to doc_lengths
# used in the BM25 algorithm
def map_doc_lengths(csv_file_line):
    csv_file_line_elements = csv_file_line.split('\",\"')
    doc_id = re.sub("[^A-Za-z0-9 ]", "", csv_file_line_elements[0])
    words_for_doc = list(set(re.sub("[^A-Za-z0-9 ]", "", csv_file_line_elements[1]).lower().split(' ')))
    return (doc_id, len(words_for_doc))

def parse_queries(query_rdd_element):
    query_rdd_elements = query_rdd_element.split('\",\"')
    query_id = re.sub("[^A-Za-z0-9 ]", "", query_rdd_elements[0])
    query_terms = re.sub("[^A-Za-z0-9 ]", "", query_rdd_elements[1]).split(" ")
    return (query_id, query_terms)

def parse_qrels(qrels_rdd_element):
    qrels_rdd_elements = qrels_rdd_element.split('\",\"')
    doc_id = re.sub("[^A-Za-z0-9 ]", "", qrels_rdd_elements[0])
    query_id = doc_id = re.sub("[^A-Za-z0-9 ]", "", qrels_rdd_elements[1])
    return (doc_id, query_id)

def get_list_element_idx_0(rdd_element):
    return rdd_element[0]

def get_list_element_idx_1(rdd_element):
    return rdd_element[1]

def get_list_element_idx_2(rdd_element):
    return rdd_element[2]

def get_list_element_idx_3(rdd_element):
    return rdd_element[3]

def get_list_element_idx_4(rdd_element):
    return rdd_element[4]

def get_list_element_idx_5(rdd_element):
    return rdd_element[5]

def run_bm_25_and_qrel_eval(query_rdd_element):
    mysql_connection = mysql.connector.connect(user='root', password='root', database='wikipedia_docs')
    recall_for_query = dict()
    query_id = str(query_rdd_element[0])
    cursor = mysql_connection.cursor()
    mysql_query = "select term, JSON_EXTRACT(postings, \"$.postings\") as posting_enc from wikipedia_vocabulary_to_posting_lookup where term in ("
    for i in range(0, len(query_rdd_element[1]) - 1):
        mysql_query = mysql_query + "\"" + query_rdd_element[1][i] + "\","
    mysql_query = mysql_query + "\"" + query_rdd_element[1][-1] + "\");"
    cursor.execute(mysql_query)
    term_and_postings_dict = dict()
    for (term, posting_enc) in cursor:
        term_and_postings_dict[str(term)] = dict(pickle.loads(base64.b64decode(str(posting_enc).encode('utf-8'))))
    cursor.close()
    doc_ids = list()
    for postings in term_and_postings_dict.values():
        doc_ids.extend(list(postings.keys()))
    doc_ids = list(set(int(doc_id) for doc_id in doc_ids))
    mysql_query = "select doc_id, doc_length from wikipedia_doc_lengths where doc_id in ("
    for i in range(0, len(doc_ids)-1):
        mysql_query = mysql_query + str(doc_ids[i]) + ", "
    mysql_query = mysql_query + str(doc_ids[-1]) + ");"
    cursor = mysql_connection.cursor()
    cursor.execute(mysql_query)
    doc_ids_and_lengths_list = dict()
    for (doc_id, doc_length) in cursor:
        doc_ids_and_lengths_list[int(doc_id)] = float(doc_length)
    cursor.close()
    mysql_query = "select count(*) as count, avg(doc_length) as average from wikipedia_doc_lengths;"
    cursor = mysql_connection.cursor()
    cursor.execute(mysql_query)
    N = 1
    doc_length_average = 1
    for (count, average_length) in cursor:
        N = int(count)
        doc_length_average = float(average_length)
    cursor.close()
    k = 1.5
    b = 0.75
    doc_ids_and_rankings = dict()
    for doc_id in doc_ids:
        for query_term in query_rdd_element[1]:
            term_postings = term_and_postings_dict.get(query_term)
            term_frequency_per_doc_in_corpus = len(term_postings) if term_postings is not None else 1
            term_frequency_in_doc = term_postings.get(doc_id) if term_postings is not None else None
            term_frequency_in_doc = term_frequency_in_doc if term_frequency_in_doc is not None else 1
            doc_length = doc_ids_and_lengths_list.get(doc_id)
            doc_length = doc_length if doc_length is not None else 1
            bm25_summand = (np.log(N / term_frequency_per_doc_in_corpus) *
                            ((k + 1) * term_frequency_in_doc) /
                            (k * ((1 - b) + b * (doc_length / doc_length_average)) + term_frequency_in_doc))
            current_value = doc_ids_and_rankings.get(doc_id)
            current_value = current_value if current_value is not None else 0
            doc_ids_and_rankings[doc_id] = current_value + bm25_summand
    doc_ids_sorted_by_rank = sorted(doc_ids_and_rankings.items(), key=lambda item: item[1])
    mysql_query = "select doc_id, query_id from wikipedia_qrels where query_id = " + str(query_id) + ";"
    cursor = mysql_connection.cursor()
    cursor.execute(mysql_query)
    doc_ids_for_query = list()
    for (doc_id, query_id_from_cursor) in cursor:
        doc_ids_for_query.append(int(doc_id))
    cursor.close()
    recall_for_query[query_id] = [
        0, 0, 0, 0, 0, 0
    ]
    for doc_id in doc_ids_for_query:
        recall_for_query[query_id][0] = recall_for_query[query_id][0] + int(doc_id in doc_ids_sorted_by_rank[0:1])
        recall_for_query[query_id][1] = recall_for_query[query_id][1] + int(doc_id in doc_ids_sorted_by_rank[0:5])
        recall_for_query[query_id][2] = recall_for_query[query_id][2] + int(doc_id in doc_ids_sorted_by_rank[0:10])
        recall_for_query[query_id][3] = recall_for_query[query_id][3] + int(doc_id in doc_ids_sorted_by_rank[0:50])
        recall_for_query[query_id][4] = recall_for_query[query_id][4] + int(doc_id in doc_ids_sorted_by_rank[0:100])
        recall_for_query[query_id][5] = recall_for_query[query_id][5] + int(doc_id in doc_ids_sorted_by_rank[0:1000])
    recall_for_query[query_id][0] = recall_for_query[query_id][0] / len(doc_ids_for_query)
    recall_for_query[query_id][1] = recall_for_query[query_id][1] / len(doc_ids_for_query)
    recall_for_query[query_id][2] = recall_for_query[query_id][2] / len(doc_ids_for_query)
    recall_for_query[query_id][3] = recall_for_query[query_id][3] / len(doc_ids_for_query)
    recall_for_query[query_id][4] = recall_for_query[query_id][4] / len(doc_ids_for_query)
    recall_for_query[query_id][5] = recall_for_query[query_id][5] / len(doc_ids_for_query)
    mysql_connection.close()
    return recall_for_query.items()

for spark_conf_holder in spark_conf_holders:
    log_file_name = f"{log_directory}/{repr(spark_conf_holder)}-{int(time.time())}.log"
    if remote_execution:
        storage_level = StorageLevel(True, False, False, False, 3)
        spark_context = SparkContext(conf=spark_conf_holder.get_conf())
    else:
        storage_level = StorageLevel(True, True, False, False, 3)
        spark_context = SparkContext(master=spark_host, conf=spark_conf_holder.get_conf())
    spark_session = SparkSession(spark_context)
    # load dataset from CSV file to RDD
    train_docs_index_rdd = spark_context.textFile(f"{dataset_directory}/{training_dataset_filename}_docs")
    if repartition_rdds:
        train_docs_index_rdd = train_docs_index_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
    train_docs_index_rdd.persist(storage_level)

    # map CSV file to vocabulary-document-id pairs, flattening pairs across documents
    elements = train_docs_index_rdd.take(10)
    index_map_start_time = time.time()
    train_docs_index_rdd = train_docs_index_rdd.flatMap(inverted_index_map_function)
    if repartition_rdds:
        train_docs_index_rdd = train_docs_index_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
    train_docs_index_rdd.persist(storage_level)
    elements = train_docs_index_rdd.take(10)
    index_map_end_time = time.time()
    output_line = f"\"{repr(spark_conf_holder)}\", \"index_map_execution_time\", \"{index_map_end_time - index_map_start_time}\""
    log_file = open(log_file_name, "a")
    log_file.write(output_line + "\n")
    log_file.close()
    print(output_line)
    print(f"train_docs_index_rdd.getNumPartitions(): {train_docs_index_rdd.getNumPartitions()}")
    print(f"elements: {str(elements)[0:1000]}")

    elements = train_docs_index_rdd.take(10)
    index_reduce_start_time = time.time()
    train_docs_index_rdd = train_docs_index_rdd.reduceByKey(inverted_index_reduce_function)
    if repartition_rdds:
        train_docs_index_rdd = train_docs_index_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
    train_docs_index_rdd.persist(storage_level)
    elements = train_docs_index_rdd.take(10)
    index_reduce_end_time = time.time()
    output_line = f"\"{repr(spark_conf_holder)}\", \"index_reduce_execution_time\", \"{index_reduce_end_time - index_reduce_start_time}\""
    log_file = open(log_file_name, "a")
    log_file.write(output_line + "\n")
    log_file.close()
    print(output_line)
    print(f"train_docs_index_rdd.getNumPartitions(): {train_docs_index_rdd.getNumPartitions()}")
    print(f"elements: {str(elements)[0:1000]}")

    if run_bm25_evaluation:
        elements = train_docs_index_rdd.take(10)
        index_pickle_start_time = time.time()
        train_docs_index_rdd = train_docs_index_rdd.map(inverted_index_pickle_function)
        if repartition_rdds:
            train_docs_index_rdd = train_docs_index_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        train_docs_index_rdd.persist(storage_level)
        elements = train_docs_index_rdd.take(10)
        index_pickle_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"index_pickle_execution_time\", \"{index_pickle_end_time - index_pickle_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)
        print(f"train_docs_index_rdd.getNumPartitions(): {train_docs_index_rdd.getNumPartitions()}")
        print(f"elements: {str(elements)[0:1000]}")

        index_store_start_time = time.time()
        train_docs_index_df = spark_session.createDataFrame(train_docs_index_rdd, schema=["term", "postings"])
        train_docs_index_df.write.jdbc(
            url=mysql_connection_url,
            table="wikipedia_vocabulary_to_posting_lookup",
            properties={
                "user": "root",
                "password": "root",
                "driver": "com.mysql.cj.jdbc.Driver"
            },
            mode="overwrite"
        )
        index_store_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"index_store_execution_time\", \"{index_store_end_time - index_store_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)

        train_docs_len_map_rdd = spark_context.textFile(f"{dataset_directory}/{training_dataset_filename}_docs")
        if repartition_rdds:
            train_docs_len_map_rdd = train_docs_len_map_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        train_docs_len_map_rdd.persist(storage_level)
        elements = train_docs_len_map_rdd.take(10)
        doc_len_map_start_time = time.time()
        train_docs_len_map_rdd = train_docs_len_map_rdd.map(map_doc_lengths)
        if repartition_rdds:
            train_docs_len_map_rdd = train_docs_len_map_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        train_docs_len_map_rdd.persist(storage_level)
        elements = train_docs_len_map_rdd.take(10)
        doc_len_map_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"doc_length_map_execution_time\", \"{doc_len_map_end_time - doc_len_map_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)
        print(f"train_docs_len_map_rdd.getNumPartitions(): {train_docs_len_map_rdd.getNumPartitions()}")
        print(f"elements: {str(elements)[0:1000]}")

        doc_len_store_start_time = time.time()
        train_docs_len_map_rdd = spark_session.createDataFrame(train_docs_len_map_rdd, schema=["doc_id", "doc_length"])
        train_docs_len_map_rdd.write.jdbc(
            url=mysql_connection_url,
            table="wikipedia_doc_lengths",
            properties={
                "user": "root",
                "password": "root",
                "driver": "com.mysql.cj.jdbc.Driver"
            },
            mode="overwrite"
        )
        doc_len_store_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"doc_length_store_execution_time\", \"{doc_len_store_end_time - doc_len_store_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)

        train_qrels_rdd = spark_context.textFile(f"{dataset_directory}/{training_dataset_filename}_qrels")
        if repartition_rdds:
            train_qrels_rdd = train_qrels_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        train_qrels_rdd.persist(storage_level)
        elements = train_qrels_rdd.take(10)
        qrels_parse_start_time = time.time()
        train_qrels_rdd = train_qrels_rdd.map(parse_qrels)
        if repartition_rdds:
            train_qrels_rdd = train_qrels_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        train_qrels_rdd.persist(storage_level)
        elements = train_qrels_rdd.take(10)
        qrels_parse_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"qrels_parse_execution_time\", \"{qrels_parse_end_time - qrels_parse_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)
        print(f"train_qrels_rdd.getNumPartitions(): {train_qrels_rdd.getNumPartitions()}")
        print(f"elements: {str(elements)[0:1000]}")

        qrels_store_start_time = time.time()
        train_qrels_df = spark_session.createDataFrame(train_qrels_rdd, schema=["doc_id", "query_id"])
        train_qrels_df.write.jdbc(
            url=mysql_connection_url,
            table="wikipedia_qrels",
            properties={
                "user": "root",
                "password": "root",
                "driver": "com.mysql.cj.jdbc.Driver"
            },
            mode="overwrite"
        )
        qrels_store_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"qrels_store_execution_time\", \"{qrels_store_end_time - qrels_store_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)

        queries_rdd = spark_context.textFile(f"{dataset_directory}/{training_dataset_filename}_queries")
        if repartition_rdds:
            queries_rdd = queries_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        queries_rdd.persist(storage_level)
        queries_rdd = queries_rdd.map(parse_queries)
        if repartition_rdds:
            queries_rdd = queries_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        queries_rdd.persist(storage_level)
        elements = queries_rdd.take(10)
        bm25_start_time = time.time()
        query_recall_rdd = queries_rdd.map(run_bm_25_and_qrel_eval)
        if repartition_rdds:
            query_recall_rdd = query_recall_rdd.repartition(spark_conf_holder.get_executor_count() * reparition_multiple_of_executors)
        query_recall_rdd.persist(storage_level)
        elements = query_recall_rdd.take(10)
        bm25_end_time = time.time()
        output_line = f"\"{repr(spark_conf_holder)}\", \"bm25_and_qrel_eval_execution_time\", \"{bm25_end_time - bm25_start_time}\""
        log_file = open(log_file_name, "a")
        log_file.write(output_line + "\n")
        log_file.close()
        print(output_line)
        print(f"queries_rdd.getNumPartitions(): {queries_rdd.getNumPartitions()}")
        print(f"elements: {str(elements)[0:1000]}")

        query_recall_rdd_0 = query_recall_rdd.map(get_list_element_idx_0)
        query_recall_rdd_1 = query_recall_rdd.map(get_list_element_idx_1)
        query_recall_rdd_2 = query_recall_rdd.map(get_list_element_idx_2)
        query_recall_rdd_3 = query_recall_rdd.map(get_list_element_idx_3)
        query_recall_rdd_4 = query_recall_rdd.map(get_list_element_idx_4)
        query_recall_rdd_5 = query_recall_rdd.map(get_list_element_idx_5)

        recall_at_1 = query_recall_rdd_0.mean()
        recall_at_5 = query_recall_rdd_1.mean()
        recall_at_10 = query_recall_rdd_2.mean()
        recall_at_50 = query_recall_rdd_3.mean()
        recall_at_100 = query_recall_rdd_4.mean()
        recall_at_1000 = query_recall_rdd_5.mean()

        recall_output = (f"\"{repr(spark_conf_holder)}\", \"recall_at_1: {recall_at_1}\", \"recall_at_5: {recall_at_5}\", "
                         + f"\"recall_at_10: {recall_at_10}\", \"recall_at_50: {recall_at_50}\", "
                         + f"\"recall_at_100: {recall_at_100}\", \"recall_at_1000: {recall_at_1000}\"")
        log_file.write(recall_output)
        print(recall_output)

    spark_context.stop()


print("program complete")
