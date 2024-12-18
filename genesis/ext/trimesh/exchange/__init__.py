"""
trimesh/exchange
----------------

Contains the importers and exporters for various mesh formats.

Note that *you should probably not be using these directly*, if
you call `trimesh.load` it will then call and wrap the result
of the various loaders:

```
mesh = trimesh.load(file_name)
```
"""
