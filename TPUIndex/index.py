import numpy as np
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print('Failed initializing TPU, cannot proceed')


class Index:
    def __init__(self, vectors, worker):
        self.vectors = vectors
        self.worker = worker
        print('Building index with {} vectors on {}'.format(
            vectors.shape[0], worker))

    @tf.function
    def search(self, query_vector, top_k=20):
        with tf.device(self.worker):
            dot_product = tf.reduce_sum(tf.multiply(
                self.vectors, query_vector), axis=1)
            distances = 1 - dot_product
            sorted_indices = tf.argsort(distances)
            nearest_distances = tf.gather(distances, sorted_indices)
            return nearest_distances[:top_k], sorted_indices[:top_k]
        # ToDo: Add search using other distance metrics


class TPUIndex:
    def __init__(self, num_tpu_cores=8):
        self.workers = ['/job:worker/replica:0/task:0/device:TPU:' + str(i)
                        for i in range(num_tpu_cores)]
        self.indices = [None] * num_tpu_cores
        self.normalized_vectors = False

    def create_index(self, vectors, normalize=True):
        self.vectors = vectors
        self.normalized_vectors = normalize

        drop = self.vectors.shape[0] % len(self.workers)
        self.vecs_per_index = self.vectors.shape[0] // len(self.workers)
        self.vectors = self.vectors[:-drop]
        self.vectors = np.split(self.vectors, len(self.workers), axis=0)

        for i in range(len(self.workers)):
            worker = self.workers[i]
            with tf.device(worker):
                vecs = self.vectors[i]
                if self.normalized_vectors:
                    vecs = tf.math.l2_normalize(vecs, axis=1)
                self.indices[i] = Index(vecs, worker)

    def search(self, xq, distance_metric='cosine', top_k=10):
        dims = xq.shape

        assert len(dims) == 2, \
            '''Expected query_vector to have 2 dimesions but
               found {}'''.format(len(dims))

        assert dims[0] == 1, \
            '''Expected query_vector to have shape == [1, None]
               but got'''.format(dims)

        if distance_metric == 'cosine':
            assert self.normalized_vectors, \
                '''Currently only normalized vectors are supported
                   for searching with cosine distances'''

        Dx, Ix = [], []
        for i in range(len(self.workers)):
            print('Search running on {}'.format(self.indices[i].worker))
            d, idx = self.indices[i].search(xq, top_k)
            Dx.extend(d.numpy())
            Ix.extend(i * self.vecs_per_index + idx.numpy())

        # ToDo: Dont sort again, merge the already sorted distance arrays
        id_sorted = np.argsort(Dx)[:top_k]
        Dx = np.array(Dx)[id_sorted]
        Ix = np.array(Ix)[id_sorted]
        return Dx, Ix
