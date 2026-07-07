"""I/O and raw pixel/array plumbing.

* :mod:`image_io` - PIL <-> tensor <-> base64.
* :mod:`image_ops` - bbox / crop / dilate.
* :mod:`checkpoints` - remote model asset downloads.

No business logic here. Anything that touches disk, network, or GPU
belongs in this layer.
"""
