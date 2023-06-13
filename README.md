## Form factors ofblock copolymer micelles with spherical, ellipsoidal and cylindrical cores

This repository contains python code to compute form factors of micellese described in the following paper:

```bib
@article{pedersen2000form,
  title={Form factors of block copolymer micelles with spherical, ellipsoidal and cylindrical cores},
  author={Pedersen, Jan Skov},
  journal={Journal of Applied Crystallography},
  volume={33},
  number={3},
  pages={637--640},
  year={2000},
  publisher={International Union of Crystallography}
}
```

There two ways you can use this code:
1. Using the popular `sasmodels` : See the example in [sasmodels.ipynb](/sasmodels.ipynb)
2. Using seperate notebooks for each one of them : see the examples in [notebooks folder](/notebooks)

If you are interested in fitting these models using the `bumps` interface of `sasmodels`, see the example in [optimize.py](/optimize.py)

### Known issues:
1. Form factors of worm-like and vesicle-shaped micelles doesn't work as expected.

### To-Do : 



### Typical troubleshooting : 

1. Issues with data loader:
Primary reseaon for this is the consistency between what you see on sasmodels github vs what is actually on the PyPI. The github version moved on to using the standalone `sasdata` package while the PyPI install still uses `sas.sascalc`. So instead of doing:
```python
sasmodels.data.load_data('your_file_name.ext')
```
You should first install `sasdata` using `pip install sasdata` and then do the following:
```python
from sasdata.dataloader.loader import Loader
loader_module = Loader()
loaded_data_sets = loader_module.load('your_file_name.ext')
```
I am not sure if you are using the installation of sasview directly, this is not a problem at all.