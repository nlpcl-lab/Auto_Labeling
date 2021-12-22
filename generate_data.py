import json, copy, yaml, logging, argparse
import jsonlines
from utils import DictObj
from typing import Dict

class Query:
    def __init__(self, configs):
        self.dataset = configs.input.dataset
        save_path = configs.input.save_path.format(self.dataset)
        gen_result = configs.input.gen_result.format(self.dataset)
        self.output_path = configs.output.query.format(configs.output.dataset)

        self.create_id_and_text(save_path, gen_result)
        self.create_queries()

    def create_id_and_text(self,p1,p2):
        self.queries = dict()
        with open(p1,'r') as f:
            id_lines = f.readlines()
        with open(p2,'r') as f:
            text_lines = f.readlines()

        for id_line, text_line in zip(id_lines,text_lines):
            _id = id_line.split('\t')[1].replace("labels:","")
            _text = text_line.split('\t')[1].replace('PREDICTION:',"")
            self.queries[_id] = {'text':_text}

        print(self.queries["0"])

    def create_queries(self):
        output_data = []
        with jsonlines.open(self.output_path) as read_file:
            for line in read_file.iter():
                if line['_id'] in self.queries.keys():
                    line['text'] = self.queries[line['_id']]['text']
                output_data.append(line)

        with open(self.output_path,'w') as f:
            for l in output_data:
                json.dump(l, f)
                f.write('\n')




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate Data')
    parser.add_argument('--configs', '-cf', type=str, default='./conf/Context_Parse.yaml')

    args = parser.parse_args()
    config_filepath = args.configs

    with open(config_filepath) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    query = Query(configs)