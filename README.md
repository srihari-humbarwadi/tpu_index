## TPU Index

TPU Index is a package for fast similarity search over large collections of high dimension vectors on TPUs.
This package was built to support our project that we developed for https://tfworld.devpost.com/.

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

### ToDo:
 - [ ] Add more distance metrics
 - [ ] Optional GPU support
