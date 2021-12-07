from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random

def log_dic(d):
    for key, value in d.items():
        logging.info("{} : {}".format(key,value))


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "nq"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
data_path = os.path.join(out_dir, dataset)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

model = DRES(models.SentenceBERT("/home/tjrals/beir/output/bert-base-uncased-v2-nq"))
retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

logging.info("Result : ndcg")
log_dic(ndcg)

logging.info("Result : _map")
log_dic(_map)

logging.info("Result : recall")
log_dic(recall)

logging.info("Result : precision")
log_dic(precision)

logging.info("Result : mrr")
log_dic(mrr)

# #### Print top-k documents retrieved ####
#
# query_id, ranking_scores = random.choice(list(results.items()))
# scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# logging.info("Query : %s\n" % queries[query_id])
#
# for rank in [0,4,9,19,49,99]:
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
