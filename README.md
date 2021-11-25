# naunet_cuda_primordial

This is one example of [naunet](https://github.com/appolloford/naunet). In this example, `cusparse` solver in SUNDIALS is used to solve primordial chemical network.

## Requirements

- cmake >= 3.18
- sundials >= 5.7
- CUDA >= 10.0

## Quick Start

```
~ $ git clone https://github.com/appolloford/naunet_cuda_primordial.git
~ $ cd naunet_cuda_primordial
~ $ cmake -S. -Bbuild -DSUNDIALS_DIR=/path/to/sundials/lib/cmake/sundials \
          -DCMAKE_INSTALL_PREFIX=./ -DCMAKE_BUILD_TYPE=Release \
          -DMAKE_SHARED=ON -DMAKE_STATIC=ON -DMAKE_PYTHON=ON
~ $ cmake --build build
~ $ cd build && ctest
```

Check the results in the `build/test/`. The output should be saved in 
`evolution_singlegrid.txt` and `evolution_pymodule.txt`

