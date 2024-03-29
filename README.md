# ⚡ ThunderPack

_Blazingly fast multi-modal data format for training deep neural networks_

## ❓ TL;DR

Most deep learning comprise of media (image, video, sound) are distributed as plain individual files, but this can incur in a lot of inefficiencies due to the filesystem behaving poorly when many files are involved. ThunderPack solves the issue by using **LMDB** an lightweight database, that supports blazingly fast reads.

![](https://github.com/JJGO/thunderpack/blob/assets/read-time.png)
_Benchmark of random read access on various data tensor storage solutions (lower read time is better). ThunderPack scales very well to large amount of datapoints, whereas other solutions present significant slowdowns even when using local SSD hardware._

## 🌟 Features

- **Optimized read support** - ThunderPack offers specialized read-only APIs that are orders of magnitude faster than common generic interfaces that also support writing.
- **Concurrency Support** - ThunderPack is thread safe and can be used with `torch.data.util.DataLoader` without issue.
- **Memory mapping** - Thanks to the use of LMDB, data is memory-mapped by default, drastically reducing the read times for entries already present in the filesystem cache.
- **Data Locality** - Data and labels are stored together, reducing read times.
- **Immense Flexibility** - Unlike other dataloader formats that trade off speed with usability, ThunderPack keeps a human-friendly dictionary-like interface that supports arbitrary sampling and that can be edited after creation.
- **Improved Tensor I/O** - Faster primitives for (de)serializing Numpy `ndarray`s, Torch `Tensor`s and `Jax` `ndarray`s.
- **Extensive extension support** - ThunderPack supports a wide variety of data formats out of the box.
- **Customizability** - ThunderPack can easily be extended to support other  (also feel free to submit a PR with your extension of choice).
<!-- - **Cloud Native** - Compatible with streaming data schemes, and with built-in sharding support. -->

<!-- ## 🚀 Quickstart -->

## 💾 Installation

<!--
ThunderPack can be installed via `pip`. For the stable version:

```shell
pip install thunderpack
```
-->
Or for the latest version:

```shell
pip install git+https://github.com/JJGO/thunderpack.git
```

You can also **manually** install it by cloning it, installing dependencies, and adding it to your `PYTHONPATH`


```shell
git clone https://github.com/JJGO/thunderpack
python -m pip install -r ./thunderpack/requirements.txt

export PYTHONPATH="$PYTHONPATH:$(realpath ./thunderpack)"
```

## Quickstart

Thunderpack has asymetric APIs for writing and reading data to maximize read throughput and speed.

First, we create a dataset using the `ThunderDB` object which behaves like a dictionary and it will automatically and 
transparently encode data when assigning values. Keys are completely arbitrary and schema is left to the user. 
In this case we store the `metadata`, `samples` and all the keys corresponding to datapoints 

```python
from thunderpack import ThunderDB

with ThunderDB.open('/tmp/thunderpack_test', 'c') as db:
    db['metadata'] = {'version': '0.1', 'n_samples': 100}
    
    keys = []
    for i in range(100):
        key = f'sample{i:02d}'
        x = np.random.normal(size=(128,128))
        y = np.random.normal(size=(128,128))
        # Thunderpack will serialize the tuple and numpy arrays automatically
        db[key] = (x, y) 
        keys.append(key)
    db['samples'] = keys
```

Once created, we can read the data using `ThunderReader`, which as a dict-like API

```python
from thunderpack import ThunderReader

reader = ThunderReader('/tmp/thunderpack_test')
print(reader['metadata'])
# {'version': '0.1', 'n_samples': 100}
print(reader['samples'][:5])
# ['sample00', 'sample01', 'sample02', 'sample03', 'sample04']
print(reader['sample00'][0].shape)
# (128, 128)
```

Thunderpack provides a PyTorch compatible Dataset object via `ThunderDataset`, which 
 assigns a `._db` attribute with the `ThunderReader` object 

```python
from thunderpack.torch import ThunderDataset
class MyDataset(ThunderDataset):
    
    def __init__(self, file):
        super().__init__(file)
        # Access through self._db attribute
        self.samples = self._db['samples']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self._db[self.samples[idx]]

d = MyDataset('/tmp/thunderpack_test')
print(len(d))
# 100
print(d[0][0].shape)
# (128, 128)
```


## 📁 Supported Formats

ThunderPack supports a wide range of data formats out of the box

|  | Modality | Supported Formats |
| :-: | :-- | :-- |
| 🧮 | Tensor | npy, npz, pt, safetensors |
| 📷 | Image | jpeg, png, bmp, webp |
| 🎧 | Audio | wav, flac, ogg, mp3 |
| 🗂️ | Tabular | csv, parquet, feather, jsonl |
| 📄 | Documents | json, yaml, msgpack, txt |
| 🗜️ | Compression | lz4, zstd, gzip, bz2, snappy, brotli |
| 🧸 | Object | pickle |


## ↔ Type-Format mappings

ThunderPack automatically maps common Python data types to efficient data formats

| Type | Format |
|:-- | :--: |
| `PIL.Image` | PNG or JPEG |
| `pandas.DataFrame` | Parquet |
| `np.ndarray`, `torch.Tensor` | NumpyPack (LZ4) |
| `bool`, `int`, `float`, `complex`, `str` | MessagePack (LZ4) |
| `list`, `dict`, `tuple` | ThunderPack (LZ4) |


<!-- ## Performance Benchmarks

>>> Compare loading times of Miniplaces, OxfordFlowers, ImageNet, OASIS3d

## Tutorial

#### 1. Writing a dataset

#### 2. Reading a dataset

#### 3. Creating a PyTorch wrapper

#### 4. Defining a custom format  -->

## ✍️ Citation

```
@misc{ortiz2023thunderpack,
    author = {Jose Javier Gonzalez Ortiz},
    title = {The ThunderPack Data Format},
    year = {2023},
    howpublished = {\\url{<https://github.com/JJGO/thunderpack/>}},
}
```
