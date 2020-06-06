import os
import argparse
import math
import random
from collections import defaultdict
def load_data(data_dir, data_type="train"):
    data_t = defaultdict(list)
    with open(os.path.join(data_dir, data_type + '.txt'), "r", encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            quadruple = line.strip().split("\t")
            data_t[quadruple[3]].append(quadruple)
            # if 'SimplE' not in self.score_function_name and not self.is_visua:
            #     data += [[i[2], i[1] + "_reversed", i[0], i[3]] for i in data]
    return data_t

def get_relations(data):
    relations = sorted(list(set([d[1] for d in data])))
    return relations

def get_entities(data):
    entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
    return entities

def writing_stat(data_dir):
    data_all = []
    for data_type in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, data_type + '.txt'), "r", encoding='utf-8') as f:
            data = f.readlines()
            data = [line.strip().split("\t") for line in data]  # only cut by "\t", not by white space.
            data_all += data
    entities = get_entities(data_all)
    relations = get_relations(data_all)

    dir_name = args.dataset + "_downsampling"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name, "stat.txt")
    file = open(file_path, "w")
    file.write(str(len(entities)) + '\t' + str(len(relations)) + '\t' + str(0))
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ICEWS0515_forecasting", nargs="?", help="Which dataset to use.")
    parser.add_argument("--sampling_factor", type=float, default=0.2)
    args = parser.parse_args()

    for subset in ["train", "valid", "test"]:
        data_t = load_data(data_dir=args.dataset, data_type=subset)
        dir_name = args.dataset + "_downsampling"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_path = os.path.join(dir_name, subset + ".txt")
        file = open(file_path, "w")
        num_quad = 0
        for timestamp in list(data_t.keys()):
            subggraph_t = data_t[timestamp]
            for quadruple in subggraph_t:
                if random.random() < args.sampling_factor:
                    file.write(str(quadruple[0]) + '\t' +  str(quadruple[1])  + '\t' + str(quadruple[2]) + "\t" + str(quadruple[3])
                       + "\t" + str(-1) + '\n')
                    num_quad +=1
        file.close()
        print("num of quadurples in " + subset + " :" + str(num_quad) + '\n')

    writing_stat(dir_name)

