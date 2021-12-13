
from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, gzip
import logging
from torch.optim import Adam
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

    dataset = configs.train.data
    logging.info("Dataset is {}".format(dataset))
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
    data_path = os.path.join(out_dir, dataset)

    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    train_batch_size = 32           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
    max_seq_length = 256            # Max length for passages. Increasing it, requires more GPU memory (O(n^4))

    #### Provide any sentence-transformers or HF model
    model_name = "bert-base-uncased"
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Provide a high batch-size to train better with triplets!
    retriever = TrainRetriever(model=model, batch_size=train_batch_size)

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
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}".format(model_name, dataset))
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