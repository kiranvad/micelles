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
1. Using the popular sasmodels : See the example in [sasmodels.ipynb](/sasmodels.ipynb)
2. Using seperate notebooks for each one of them : see the examples in [notebooks folder](/notebooks)

### Known issues:
1. For some reason, the code to compute cylindrical core micelle behaves unexpectedly when using sasmodels
2. Form factors of worm-like and vesicle-shaped micelles doesn't work as expected.
3. Polydispertisty parameters are not being properly processed when using sasmodels.

### To-Do : 
1. Implement example of optimization using bumps or other similar software