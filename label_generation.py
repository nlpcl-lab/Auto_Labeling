from transformers import BartForConditionalGeneration, BartTokenizer
from wiki_extract import Wiki_Extract
import argparse
import re

p = re.compile(r'__[a-zA-Z\s]+_')

class WikiTable:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._parse()
        self.wiki = Wiki_Extract()

    def _parse(self):
        def _preprocess(t: str):
            result = {}
            a = [j.replace("_","") for j in p.findall(t)]
            v = [j.strip() for j in re.split(r'__[a-zA-Z\s]+_', t)[1:]]
            assert len(a)==len(v)
            for i in range(len(a)):
                if a[i] in result.keys():
                    result[a[i]].append(v[i])
                else:
                    result[a[i]] = [v[i]]
            return result
        context = []
        table = []
        with open(self.file_path,'r') as f:
            _file = f.readlines()
        for line in _file:
            if 'CONTEXT:' in line:
                context.append(line.replace('CONTEXT:','').replace('\n','').strip())
            elif 'PREDICTION:' in line:
                table.append(line.replace('PREDICTION:','').replace('\n','').strip())
        assert len(context) == len(table)
        return [(context[i],_preprocess(table[i])) for i in range(len(context))]

    def _collect_key(self):
        result = []
        for i in self.data:
            print(i)
            result = result + list(i[1].keys())
            print(result)
            exit(0)
        print(set(result))






if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--file_path','-fp',type=str, default='/home/tjrals/beir/data_output/dummy/result')

    args=parser.parse_args()
    file_path = args.file_path
    wikitable = WikiTable(file_path)
    print(wikitable.data[10])
    wikitable._collect_key()