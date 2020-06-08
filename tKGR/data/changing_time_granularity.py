import os
import argparse
import math
import shutil

def load_data(data_dir, data_type="train"):
    with open(os.path.join(data_dir, data_type + '.txt'), "r", encoding='utf-8') as f:
        data = f.readlines()
        data = [line.strip().split("\t") for line in data]  # only cut by "\t", not by white space.
        # if 'SimplE' not in self.score_function_name and not self.is_visua:
        #     data += [[i[2], i[1] + "_reversed", i[0], i[3]] for i in data]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ICEWS0515_forecasting", nargs="?", help="Which dataset to use.")
    parser.add_argument("--increasing_factor", type=float, default=10)
    parser.add_argument("--time_granularity", type=float, default=24)
    args = parser.parse_args()

    for subset in ["train", "valid", "test"]:
        data= load_data(data_dir=args.dataset, data_type=subset)
        dir_name = args.dataset + "_" + str(args.time_granularity * args.increasing_factor)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, subset + ".txt")

        file = open(file_path, "w")
        for quadruple in data:
            file.write(str(quadruple[0]) + '\t' +  str(quadruple[1])  + '\t' + str(quadruple[2]) + "\t" +
                str(math.floor(float(quadruple[3])/(args.time_granularity * args.increasing_factor))*(args.time_granularity * args.increasing_factor))
                       + "\t" + str(-1) + '\n')
        file.close()

    shutil.copy2("ICEWS0515_forecasting/entity2id.txt", dir_name)
    shutil.copy2("ICEWS0515_forecasting/relation2id.txt", dir_name)