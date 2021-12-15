from transformers import BartForConditionalGeneration, BartTokenizer
from wiki_extract import Wiki_Extract
import argparse
import re
from tqdm import tqdm
import pickle

p = re.compile(r'__[a-zA-Z\s]+_')




class WikiTable:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.wiki = Wiki_Extract()
        self.data = self._parse()

    def _parse(self):
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
        return [(context[i],self._preprocess(table[i])) for i in tqdm(range(len(context)))]

    def _preprocess(self,t: str):
        a = [j.replace("_","") for j in p.findall(t)]
        v = [j.strip() for j in re.split(r'__[a-zA-Z\s]+_', t)[1:]]
        assert len(a)==len(v)
        result = {}
        for i in range(len(a)):
            summary = self.wiki.get_summary(v[i])
            if not summary:
                result.update({v[i]:{'attr' : a[i],'ext_sum': summary}})
                break
        return result

    def _collect_key(self):
        result = []
        for i in self.data:
            result = result + list(i[1].keys())
        return set(result)






if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--file_path','-fp',type=str, default='/home/tjrals/beir/data_output/dummy/result')

    args=parser.parse_args()
    file_path = args.file_path
    wikitable = WikiTable(file_path)
    print(wikitable.data[0])
    with open('/home/tjrals/beir/data_output/dummy/result_with_ext_info','wb') as f:
        pickle.dump(wikitable,f)
    with open('/home/tjrals/beir/data_output/dummy/result_with_ext_info','rb') as f:
        wiki_2 = pickle.load(f)
    print(wiki_2.data[0])