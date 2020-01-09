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
        vectors
        self.embeddings = tf.cast(vectors, dtype=tf.bfloat16)
        self.worker = worker
        print('Building index with {} vectors on {}'.format(
            vectors.shape[0], worker))
        
    def appendEmbeds(self, vectors):
        newEmbeds = tf.cast(vectors, dtype=tf.bfloat16)
        self.embeddings = tf.concat((self.embeddings, newEmbeds), axis=0)

    @tf.function
    def search(self, query_vector, top_k=20):
        with tf.device(self.worker):
            dot_product = tf.reduce_sum(tf.multiply(
                self.embeddings, query_vector), axis=1)
            distances = 1 - dot_product
            sorted_indices = tf.argsort(distances)
            nearest_distances = tf.cast(tf.gather(distances, sorted_indices), dtype=tf.float32)
            return nearest_distances[:top_k], sorted_indices[:top_k]
        # ToDo: Add search using other distance metrics




class TPUIndex:
    def __init__(self, num_tpu_cores=8):
        self.workers = ['/job:worker/replica:0/task:0/device:TPU:' + str(i)
                        for i in range(num_tpu_cores)]
        self.indices = [None] * num_tpu_cores
        self.normalized_vectors = False

    def create_index(self, vectors, normalize=True):
        self.normalized_vectors = normalize
        self.vecs_per_index = vectors.shape[0] // len(self.workers)

        numToAdd = vectors.shape[0] % len(self.workers)
        toAddZeros = np.zeros_like(vectors[-numToAdd:])
        vectors = np.concatenate((vectors, toAddZeros), axis=0)
        vectors = np.split(vectors, len(self.workers), axis=0)

        for i in range(len(self.workers)):
            worker = self.workers[i]
            with tf.device(worker):
                vecs = vectors[i]
                if self.normalized_vectors:
                    vecs = tf.math.l2_normalize(vecs, axis=1)
                self.indices[i] = Index(vecs, worker)

    def append_index(self, vectos, normalize=True):
        self.normalized_vectors = normalize
        self.vecs_per_index = self.vecs_per_index + vectors.shape[0] // len(self.workers)

        numToAdd = vectors.shape[0] % len(self.workers)
        toAddZeros = tf.zeros_like(vectors[-numToAdd:])
        vectors = tf.concat((vectors, toAddZeros), axis=0)
        vectors = np.split(vectors, len(self.workers), axis=0)

        for i in range(len(self.workers)):
            worker = self.workers[i]
            with tf.device(worker):
                vecs = vectors[i]
                if self.normalized_vectors:
                    vecs = tf.math.l2_normalize(vecs, axis=1)
                self.indices[i].appendEmbeds(vecs)

    def search(self, xq, distance_metric='cosine', top_k=10):
        dims = xq.shape
        xq = tf.cast(xq, dtype=tf.bfloat16)
        
        assert len(dims) == 2, \
            '''Expected xq to have 2 dimesions but
               found {}'''.format(len(dims))

        assert dims[0] == 1, \
            '''Expected xq to have shape == [1, None]
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
