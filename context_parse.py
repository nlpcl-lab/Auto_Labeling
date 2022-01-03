import spacy
import random
import argparse
from tqdm import tqdm
import yaml
import logging
import pathlib, os
from utils import DictObj
from generator.wiki_extract import Wiki_Extract
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

data_format = "__SUB__ __name__ {} __type__ {} __des__ {} " \
              "__PRE__ __name__ {} __token__ {} " \
              "__OBJ__ __name__ {} __type__ {} __des__ {}"
text_format = "text:{}\tlabels:{}\tepisode_done:True\n"

def make_text(sub,pre,obj,ids):
    data = data_format.format(sub['name'],sub['type'],sub['des'],
                              pre['name'],pre['token'],
                              obj['name'],obj['type'],obj['des'])
    text = text_format.format(data,ids)
    return text

nlp = spacy.load("en_core_web_sm")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Context Parse')
    parser.add_argument('--configs', '-cf', type=str, default='./conf/Context_Parse.yaml')

    args = parser.parse_args()
    config_filepath = args.configs
    with open(config_filepath) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    dataset = configs.input.dataset
    save_path = configs.input.save_path
    rand = configs.input.random
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    nlp = spacy.load("en_core_web_sm")
    wiki = Wiki_Extract()

    logging.info("Dataset is {}".format(dataset))
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
    data_path = os.path.join(out_dir, dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    result = []
    index = 0
    for k,v in tqdm(qrels.items()):
        v = list(v.keys())[0]
        query = queries[k]
        context = corpus[v]
        doc = nlp(context['text'])
        if len(doc.ents) != 0:
            ents = list(doc.ents)
            if rand == "True":
                print('check')
                random.shuff(ents)
            sub_type = wiki.get_categories(context['title'])
            for ent in ents:
                if ent.label_ == "DATE" or ent.label_ == 'TIME' or ent.label_ == 'CARDINAL':
                    obj_type = ""
                    obj_sum = [str(s) for s in list(doc.sents) if ent.text in str(s)][0]
                else:
                    obj_type = wiki.get_categories(ent.text)
                    obj_sum = wiki.get_summary(ent.text)
                    if obj_sum != "":
                        obj_sum = nlp(obj_sum)
                        obj_sum = list(obj_sum.sents)[0]
                    if str(obj_sum) == "" or "\n" in str(obj_sum):
                        obj_sum = [str(s) for s in list(doc.sents) if ent.text in str(s)][0]
                sub = {'name' : context['title'],
                       'type' : sub_type,
                       'des' : context['text']}
                pre = {'name' : ent.label_.lower(),
                       'token' : ""}
                obj = {'name' : ent.text,
                       "type" : obj_type,
                       "des" : obj_sum
                       }
                text = make_text(sub,pre,obj,v)
                result.append(text)
                if len(result) >= 1000:
                    print("Save_Data")
                    with open(save_path.format(dataset,index), 'w') as f:
                        for text in result:
                            f.write(text)
                        index = index + 1
                        result = []

