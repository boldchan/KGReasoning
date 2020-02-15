import os
import tKGR.data
DataDir = os.path.dirname(tKGR.data.__file__)

class Data:
    def __init__(self, dataset=None):
        # load data
        self.train_data = self.load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self.load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self.load_data(os.path.join(DataDir, dataset), "test")
        self.data = self.train_data + self.valid_data + self.test_data


        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations
                                                 if i not in self.train_relations] + [i for i in self.test_relations
                                                                                      if i not in self.train_relations]
        self.timestamps = self.get_timestamps(self.data)

        self.entity_idxs, self.relation_idxs, self.timestamp_idxs = self.get_idx()

    def load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type))) as f:
            data = f.readlines()
            data = [line.strip().split("\t") for line in data] #only cut by "\t", not by white space.
            data += [[i[2], i[1]+"_reverse", i[0], i[3]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_timestamps(self, data):
        timestamps = sorted(list(set([d[3] for d in data])))
        return timestamps

    def get_idx(self):
        entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        timestamp_idxs = {self.timestamps[i]: i for i in range(len(self.timestamps))}
        return  entity_idxs, relation_idxs, timestamp_idxs