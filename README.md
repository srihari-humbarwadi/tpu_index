[![HitCount](http://hits.dwyl.io/srihari-humbarwadi/tpu_index.svg)](http://hits.dwyl.io/srihari-humbarwadi/tpu_index)

## TPU Index

<p align="center">
  <img src="https://i.imgur.com/BJdZE21.png">
</p>

TPU Index is a package for fast similarity search over large collections of high dimension vectors on TPUs.
This package was built to support our project that we developed for https://tfworld.devpost.com/.

Uses:
1) Dealing with a large number of vectors that do not fit on a CPU. TPU v2 has 8x8=64 gbs. TPU v3 has 16x8=128 gbs. 
2) Speed up similarity searches. On a colab TPU v2, a single cos similairty search of 19.5 million vectors of dimension 512 takes ~1.017 seconds. 

Link to our project: https://devpost.com/software/naturallanguagerecommendations


### Installation
`!pip install tpu-index`


### Basic usage
```
from tpu_index import TPUIndex

index = TPUIndex(num_tpu_cores=8)
index.create_index(vectors)  # vectors = numpy array, shape == [None, None]

...
D, I = index.search(xq, distance_metric='cosine', top_k=5)
```

### For large numbers of vectors that do not fit on the CPU, add them in chunks
```
index.create_index(vectorsChunk1)  # vectors = numpy array, shape == [None, None]

for file in files:
     vectorChunk = np.load(file)
     index.append_index(vectorChunk)
     
# Now perform search 
D, I = index.search(xq, distance_metric='cosine', top_k=5)

```

### ToDo:
 - [ ] Add more distance metrics
 - [ ] Optional GPU support
