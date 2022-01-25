"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random
import logging
import argparse
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', '-d', type=str, default='nq_rev')
args = parser.parse_args()
eval_dataset = args.dataset
eval_type = 'train'

out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
data_path = os.path.join(out_dir, eval_dataset)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=eval_type)


hostname = "http://localhost:9200"
index_name = "nq"

initialize = True # False

number_of_shards = 5
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
top_k = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="top_k_acc")

result = {
    "ndcg": ndcg,
    "_map": _map,
    "mrr": mrr,
    "acc": top_k,
    "recall": recall,
    "precision": precision
}

with open(os.path.join(data_path, "{}.json".format("BM25_result")), 'w') as fw:
    json.dump(result, fw)
