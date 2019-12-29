## TPU Index

TPU Index is a package for fast similarity search over large collections of high dimension vectors on TPUs.
This package was built to support our project that we developed for https://tfworld.devpost.com/.

Link to our project: https://devpost.com/software/naturallanguagerecommendations


### Installation
`!pip install tpu_index`


### Basic usage
```
from TPUIndex.index import TPUIndex

index = TPUIndex(num_tpu_cores=8)
index.create_index(<vectors>)  # vectors.shape == [None, None]

...
D, I = index.search(xq, distance_metric='cosine', top_k=5)
```

### ToDo:
 - [ ] Add more distance metrics
 - [ ] Optional GPU support
