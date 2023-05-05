# ThunderPack

_Blazingly fast multi-modal data format for training deep neural networks_

## TL;DR

Most deep learning comprise of media (image, video, sound) are distributed as plain individual files, but this can incur in a lot of inefficiencies due to the filesystem behaving poorly when many files are involved. ThunderPack solves the issue by using **LMDB** an lightweight database, that supports blazingly fast reads.

![](https://github.com/JJGO/thunderpack/blob/assets/read-time.png)
_Benchmark of random read access on various data tensor storage solutions (lower read time is better). ThunderPack scales very well to large amount of datapoints, whereas other solutions present significant slowdowns even when using local SSD hardware._

## Features

- **Optimized read support** - ThunderPack offers specialized read-only APIs that are orders of magnitude faster than common generic interfaces that also support writing.
- **Concurrency Support** - ThunderPack is thread safe and can be used with `torch.data.util.DataLoader` without issue.
- **Memory mapping** - Thanks to the use of LMDB, data is memory-mapped by default, drastically reducing the read times for entries already present in the filesystem cache.
- **Data Locality** - Data and labels are stored together, reducing read times.
- **Immense Flexibility** - Unlike other dataloader formats that trade off speed with usability, ThunderPack keeps a human-friendly dictionary-like interface that supports arbitrary sampling and that can be edited after creation.
- **Improved Tensor I/O** - Faster primitives for (de)serializing Numpy `ndarray`s, Torch `Tensor`s and `Jax` `ndarray`s.
- **Extensive extension support** - ThunderPack supports a wide variety of data formats out of the box.
- **Customizability** - ThunderPack can easily be extended to support other  (also feel free to submit a PR with your extension of choice).
- **Cloud Native** - Compatible with streaming data schemes, and with built-in sharding support.

## Installation

ThunderPack can be installed via `pip`. For the stable version:

```shell
pip install thunderpack
```

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


<!-- ## Performance Benchmarks

>>> Compare loading times of Miniplaces, OxfordFlowers, ImageNet, OASIS3d

## Tutorial

#### 1. Writing a dataset

#### 2. Reading a dataset

#### 3. Creating a PyTorch wrapper

#### 4. Defining a custom format

## Citation -->
