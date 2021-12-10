import json, csv, os, pathlib
import argparse
from typing import Dict
from tqdm import tqdm

# class DataHandler:
# 	def __init__(self,_dir):
# 		self.passage_dir = None
# 		self.passage_dict = None
# 		self.passage_map = None
# 		self.query_dir = None
# 		self.query_dict = None


# 	def set_passage_dir(self,_passage_dir):
# 		pass

# 	@staticmethod
# 	def check_dir(dir):
# 		if os.path.exists(dir):
# 			print("{} exists".format(dir))
# 			return True
# 		else:
# 			return False

def write_to_json(output_file: str, data: Dict[str, Dict]):
	with open(output_file, 'w') as fOut:
		for idx, d in data.items():
			dump = d.copy()
			dump.update({"_id": str(idx),"metadata": {}})
			json.dump(dump, fOut)
			fOut.write('\n')

def write_to_tsv(output_file: str, data: Dict[str, str]):
    with open(output_file, 'w') as fOut:
        writer = csv.writer(fOut, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id, corpus_dict in data.items():
            for corpus_id, score in corpus_dict.items():
                writer.writerow([int(query_id), int(corpus_id), score])

def read_json(dir: str):
	with open(dir,'r') as f:
		_file = json.load(f)
	return _file


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset','-d',type=str,default='nq')
	parser.add_argument('--input_path','-ip',type=str,default='/home/tjrals/DPR/downloads/data')
	args = parser.parse_args()
	dataset_name = args.dataset
	input_path = args.input_path

	if dataset_name == 'nq':
		out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "dataset")
		dataset_dir = os.path.join(out_dir,dataset_name+'_our_1')

		if os.path.exists(dataset_dir):
			print("dataset dir already exists")
			# exit(0)
		else:
			os.makedirs(dataset_dir)

#make corpus.jsonl file
		passage_dir = [input_path, 'wikipedia_split','psgs_w100.tsv']
		passage_dir = os.path.join(*passage_dir)

		if not os.path.isfile(passage_dir):
			print(passage_dir)
			print('passage file not exists')
			exit(0)
		else:
			passage_dict = {}
			reader = csv.reader(open(passage_dir, encoding="utf-8"),delimiter="\t", quoting=csv.QUOTE_MINIMAL)
			next(reader)

			for row in tqdm(reader):
				_id, text, title = row[0], row[1], row[2]
				passage_dict[_id] = {'title':title,'text':text}

			output_passage_path = os.path.join(dataset_dir ,'corpus.jsonl')
			# if not os.path.isfile(output_passage_path):
			write_to_json(os.path.join(dataset_dir ,'corpus.jsonl'),passage_dict)

		# passage_map = {v['text']:int(k) for k,v in passage_dict.items()}

		query_dir = [input_path, 'retriever']
		query_dir = os.path.join(*query_dir)

		if not os.path.exists(query_dir):
			print(query_dir)
			print("there is no query directory")
			exit(0)
		else:
			qrels_dir = os.path.join(dataset_dir, "qrels")
			if os.path.exists(qrels_dir):
				print("{} already exists".format(qrels_dir))
				# exit(0)
			else:
				os.makedirs(qrels_dir)

			query_dict = {}

# make train tsv file
			train_dict = {}

			train_dir = os.path.join(query_dir,'nq-train-v4.json')
			if not os.path.isfile(train_dir):
				print("{} isn't correct dir.".format(train_dir))
				exit(0)

			train_dump = read_json(train_dir)

			for index,v in tqdm(enumerate(train_dump)):
				train_dict[str(index)] = {v['positive_ctxs'][0]['passage_id']:1}
				query_dict[str(index)] = {"text":v['question']}

			train_dir = os.path.join(qrels_dir, 'train.tsv')
			# if not os.path.isfile(train_dir):
			# 	print("{} alreay exists".format(train_dir))
			# else:
			write_to_tsv(train_dir,train_dict)

# make dev tsv file
			dev_dict = {}

			dev_dir = os.path.join(query_dir,'nq-dev.json')
			if not os.path.isfile(dev_dir):
				print("{} isn't correct dir.".format(dev_dir))
				exit(0)

			dev_dump = read_json(dev_dir)

			for index,v in tqdm(enumerate(dev_dump)):
				dev_dict[str(len(train_dump)+index)] = {d['passage_id']:1 for d in v['positive_ctxs']}
				query_dict[str(len(train_dump)+index)] = {"text":v['question']}

			dev_dir = os.path.join(qrels_dir, 'dev.tsv')
			# if not os.path.isfile(dev_dir):
			# 	print("{} alreay exists".format(dev_dir))
			# else:
			write_to_tsv(dev_dir,dev_dict)

# make queires.jsonl file
			output_queries_path = os.path.join(dataset_dir, 'queries.jsonl')
			# if not os.path.isfile(output_queries_path):
			write_to_json(output_queries_path,query_dict)


			import IPython; IPython.embed(); exit(1)










