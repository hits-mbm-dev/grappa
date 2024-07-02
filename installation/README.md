## DGL Installation

To train grappa on a GPU, a CUDA version of DGL needs to be installed along with torch as explained in (https://www.dgl.ai/pages/start.html), e.g. by running
```
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install pyyaml pydantic #misses in dgl-requirements
```

A common installation bug of dgl (as of 2024) is that it installs torch versions that it does not expect and thus cannot find torch-version-named .so files:
```
python -c "import dgl"

-> FileNotFoundError: Cannot find DGL C++ graphbolt library at YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.3.1.so
```

To fix this, find a compatible torch version (in the example, 2.1.0):

```
ls YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt*

-> YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.1.0.so
```

and install the torch version by hand:
```
pip install torch==2.1.0
```