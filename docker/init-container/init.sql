

create database wikipedia_docs;

create table wikipedia_docs.wikipedia_doc_lengths (doc_id INT primary key, doc_length INT);

create table wikipedia_docs.wikipedia_vocabulary_to_posting_lookup (term varchar(512), postings json);

create table wikipedia_docs.wikipedia_qrels (doc_id INT, query_id INT);

