import pathlib
import shutil
from contextlib import contextmanager
from typing import Union, Dict, Optional

import numpy as np
import torch
import pandas as pd
import PIL.Image
import PIL.JpegImagePlugin

from .formats import SupportedFormats, CompressionFormat, FileFormat


def autoencode(obj: object, filename: str) -> bytes:
    ext = str(filename).partition(".")[2]
    fmt = SupportedFormats.get_format(ext)
    data = fmt.encode(obj)
    return data


def autodecode(data: bytes, filename: str) -> object:
    ext = str(filename).partition(".")[2]
    fmt = SupportedFormats.get_format(ext)
    obj = fmt.decode(data)
    return obj
    ext = str(filename).partition(".")[2]


def autoload(path: Union[str, pathlib.Path]) -> object:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file {str(path)}")

    ext = path.suffix.strip(".")
    fmt = SupportedFormats.get_format(ext)
    return fmt.load(path)


def autosave(obj, path: Union[str, pathlib.Path], parents=True) -> object:
    if isinstance(path, str):
        path = pathlib.Path(path)
    ext = path.suffix.strip(".")
    fmt = SupportedFormats.get_format(ext)
    if parents:
        path.parent.mkdir(exist_ok=True, parents=True)
    fmt.save(obj, path)


@contextmanager
def inplace_edit(file, backup=False):
    if isinstance(file, str):
        file = pathlib.Path(file)
    obj = autoload(file)
    yield obj
    if backup:
        shutil.copy(file, file.with_suffix(file.suffix + ".bk"))
    autosave(obj, file)


_ENCODERS = {
    np.ndarray: ".msgpack.lz4",
    torch.Tensor: ".pt.lz4",
    PIL.JpegImagePlugin.JpegImageFile: ".jpg",
    PIL.Image.Image: ".png",
    pd.DataFrame: ".parquet",
    (str, list, tuple, dict, int, float, bool, bytes): ".msgpack.lz4",
    object: ".pkl.lz4",
}


class DefaultExtensions:

    _type2ext: Dict[type, str] = {
        np.ndarray: ".msgpack.lz4",
        torch.Tensor: ".pt.lz4",
        PIL.JpegImagePlugin.JpegImageFile: ".jpg",
        PIL.Image.Image: ".png",
        pd.DataFrame: ".parquet",
        int: ".msgpack",
        float: ".msgpack",
        bool: ".msgpack",
        bytes: ".bin",
        list: ".msgpack.lz4",
        tuple: ".msgpack.lz4",
        dict: ".msgpack.lz4",
    }
    _fallback = ".pkl.lz4"

    @classmethod
    def register(cls, T: type, ext: str):
        cls._type2ext[T] = ext

    @classmethod
    def set_fallback(cls, ext: str):
        cls._fallback = ext

    @classmethod
    def get_extension(cls, T: type) -> str:
        if T in cls._type2ext:
            return cls._type2ext[T]
        return cls._fallback


def autopackb(obj: object, ext: Optional[str] = None) -> bytes:
    if ext is None:
        ext = DefaultExtensions.get_extension(type(obj))
    data = autoencode(obj, ext)
    # doing ext || null || payload is faster than msgpack-ing it
    return ext.encode() + b"\x00" + data


def autounpackb(data: bytes, extension: Optional[str] = None) -> object:
    ext, _, data = data.partition(b"\x00")
    return autodecode(data, ext.decode())
