from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
import pathlib, os, gzip
import logging
from torch.optim import Adam
from tqdm import tqdm
import yaml, argparse

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
    parser.add_argument('--data','-d',type=str, default='None')

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

    if args.data == 'None':
        dataset = configs.train.data
    else:
        dataset = args.data

    version = configs.train.version
    logging.info("Dataset is {}".format(dataset))
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
    data_path = os.path.join(out_dir, dataset)

    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    if version == "Hard":
        hostname = "http://localhost:9200"
        index_name = "nq"

        #### Intialize ####
        # (1) True - Delete existing index and re-index all documents from scratch
        # (2) False - Load existing index
        initialize = True  # False

        #### Sharding ####
        # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
        # SciFact is a relatively small dataset! (limit shards to 1)
        number_of_shards = 1
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

        # (2) For datasets with big corpus ==> keep default configuration
        # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        bm25 = EvaluateRetrieval(model)

        #### Index passages into the index (seperately)
        bm25.retriever.index(corpus)

        triplets = []
        qids = list(qrels)
        hard_negatives_max = 10

        #### Retrieve BM25 hard negatives => Given a positive document, find most similar lexical documents
        for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25"):
            query_id, query_text = qids[idx], queries[qids[idx]]
            pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
            pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
            hits = bm25.retriever.es.lexical_multisearch(texts=pos_doc_texts, top_hits=hard_negatives_max + 1)
            for (pos_text, hit) in zip(pos_doc_texts, hits):
                for (neg_id, _) in hit.get("hits"):
                    if neg_id not in pos_docs:
                        neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                        triplets.append([query_text, pos_text, neg_text])


    train_batch_size = 32           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
    max_seq_length = 256            # Max length for passages. Increasing it, requires more GPU memory (O(n^4))

    #### Provide any sentence-transformers or HF model
    model_name = configs.train.bert
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Provide a high batch-size to train better with triplets!
    retriever = TrainRetriever(model=model, batch_size=train_batch_size)

    if version == "Hard":
        train_samples = retriever.load_train_triplets(triplets=triplets)
        train_dataloader = retriever.prepare_train_triplets(train_samples)
    else:
        #### Prepare triplets samples
        train_samples = retriever.load_train(corpus, queries, qrels)
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    #### Training SBERT with cosine-product
    # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    # #### training SBERT with dot-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model,scale=1.0, similarity_fct=util.dot_score)

    #### If no dev set is present from above use dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-{}-{}".format(model_name,version,dataset))
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 25
    evaluation_steps = 10000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
    adam_eps = 1e-8
    adam_betas = (0.9, 0.999)
    max_grad_norm = 2.0
    train_rolling_loss_step = 100
    learning_rate = 2e-5
    weight_decay = 0.0

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=ir_evaluator,
                  epochs=num_epochs,
                  output_path=model_save_path,
                  warmup_steps=warmup_steps,
                  evaluation_steps=evaluation_steps,
                  use_amp=True,
                  max_grad_norm = 2.0,
                  weight_decay=0.0,
                  optimizer_class=Adam,
                  optimizer_params={'lr':learning_rate,'betas':adam_betas,'eps':adam_eps},
                  scheduler='warmuplinear'
                  )