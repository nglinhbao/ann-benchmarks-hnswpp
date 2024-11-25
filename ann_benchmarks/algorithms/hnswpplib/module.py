import hnswpplib
import numpy as np
import time

from ..base.module import BaseANN


class HnswPPLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = "hnswlib (%s)" % (self.method_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswpplib.Index(self.metric, len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"], random_seed=100, allow_replace_deleted=False
        )
        data_labels = np.arange(len(X))
        prepare_start = time.time()
        self.p.prepare_index(X)
        self.prepare_time = time.time() - prepare_start
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        # self.p.set_ef(ef)
        pass

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
