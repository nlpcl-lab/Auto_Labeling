from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import yaml
import logging
import pathlib, os
import random, argparse

class DictObj:

    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--configs', '-cf', type=str, default='./conf/IR.yaml')

    args = parser.parse_args()
    config_filepath = args.configs
    with open(config_filepath) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    model_name = configs.eval.model
    train_dataset = configs.eval.train_data
    eval_dataset = configs.eval.eval_data
    logging.info("Dataset is {}".format(eval_dataset))
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
    data_path = os.path.join(out_dir, eval_dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}".format(model_name, train_dataset))
    #trained DPR model

    model = DRES(models.SentenceBERT(model_save_path))
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    top_k = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="top_k_acc")
