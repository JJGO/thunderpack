import io
import itertools
import json
import pathlib
import pickle
from abc import ABC
from typing import Any, Dict, List, Literal, Tuple

import msgpack
import msgpack_numpy as m
import numpy as np
import pandas as pd
import PIL.Image
import PIL.JpegImagePlugin
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import soundfile as sf
import yaml

import torch

m.patch()


class FileExtensionError(Exception):
    pass


class FileFormat(ABC):

    """
    Base class that other formats inherit from
    children formats must overload one of (save|encode) and
    one of (load|decode), the others will get mixin'ed
    """

    EXTENSIONS = []

    @classmethod
    def check_fp(cls, fp):
        if isinstance(fp, io.BytesIO):
            return fp
        if isinstance(fp, str):
            fp = pathlib.Path(fp)
        if fp.suffix not in cls.EXTENSIONS:
            msg = f"{cls.__name__} expects formats {cls.EXTENSIONS}, received {fp.suffix} instead"
            raise FileExtensionError(msg)
        return fp

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("wb") as f:
            f.write(cls.encode(obj))

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("rb") as f:
            return cls.decode(f.read())

    @classmethod
    def encode(cls, obj) -> bytes:
        mem = io.BytesIO()
        cls.save(obj, mem)
        return mem.getvalue()

    @classmethod
    def decode(cls, data: bytes) -> object:
        mem = io.BytesIO(data)
        return cls.load(mem)



class CompressedFileFormat(FileFormat):
    pass


class BinaryFormat(FileFormat):

    EXTENSIONS = [".bin"]

    @classmethod
    def encode(cls, obj) -> bytes:
        return obj

    @classmethod
    def decode(cls, data: bytes) -> object:
        return data


class NpyFormat(FileFormat):

    EXTENSIONS = [".npy"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.save(fp, obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return np.load(fp)


class NpzFormat(FileFormat):

    EXTENSIONS = [".npz"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.savez(fp, **obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return dict(np.load(fp))


class PtFormat(FileFormat):

    EXTENSIONS = [".pt"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        torch.save(obj, fp)

    @classmethod
    def load(cls, fp):
        fp = cls.check_fp(fp)
        return torch.load(fp)


class YamlFormat(FileFormat):

    EXTENSIONS = [".yaml", ".yml"]

    @classmethod
    def encode(cls, obj) -> str:
        return yaml.safe_dump(obj, indent=2).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> object:
        return yaml.safe_load(data)

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("w") as f:
            yaml.safe_dump(obj, f)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return yaml.safe_load(f)


class NumpyJSONEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonFormat(FileFormat):

    EXTENSIONS = [".json"]

    @classmethod
    def encode(cls, obj) -> bytes:
        return json.dumps(obj, cls=NumpyJSONEncoder).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> object:
        return json.loads(data.decode("utf-8"))

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("w") as f:
            json.dump(obj, f, cls=NumpyJSONEncoder)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return json.load(f)


class JsonlFormat(FileFormat):

    EXTENSIONS = [".jsonl"]

    @classmethod
    def save(cls, obj, fp):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        obj.to_json(fp, orient="records", lines=True)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        return pd.read_json(fp, lines=True)


class CsvFormat(FileFormat):

    EXTENSIONS = [".csv"]

    @classmethod
    def save(cls, obj, fp):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        obj.to_csv(fp, index=False)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        return pd.read_csv(fp)


# class ParquetFormat(FileFormat):

#     EXTENSIONS = [".parquet", ".PARQUET", ".pq", ".PQ"]

#     @classmethod
#     def save(cls, obj, fp):
#         if not isinstance(obj, pd.DataFrame):
#             raise TypeError("Can only serialize pd.DataFrame objects")
#         fp = cls.check_fp(fp)
#         obj.to_parquet(fp, index=False)

#     @classmethod
#     def load(cls, fp) -> object:
#         return pd.read_parquet(fp)
#         fp = cls.check_fp(fp)


class ParquetFormat(CompressedFileFormat):

    # Tweaked version of write/read_parquet to support
    # storing .attrs metadata in the parquet metadata fields

    EXTENSIONS = [".parquet", ".pq"]

    @classmethod
    def save(cls, obj, fp):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        table = pa.Table.from_pandas(obj)
        meta = json.dumps(obj.attrs)
        new_meta = {b"custom_metadata": meta, **table.schema.metadata}
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, fp)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        table = pq.read_table(fp)
        df = table.to_pandas()
        custom_meta = table.schema.metadata.get(b"custom_metadata", "{}")
        df.attrs = json.loads(custom_meta)
        return df


class FeatherFormat(CompressedFileFormat):
    # TODO feather supports lz4 and zstd natively, reject wrapper-compression
    EXTENSIONS = [".feather"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        feather.write_feather(obj, fp)

    @classmethod
    def load(cls, fp) -> object:
        return feather.read_feather(fp)


class PickleFormat(FileFormat):

    EXTENSIONS = [".pkl", ".pickle"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def encode(cls, obj) -> bytes:
        return pickle.dumps(obj)

    @classmethod
    def decode(cls, data: bytes) -> object:
        return pickle.loads(data)


class ImageFormat(CompressedFileFormat):
    @classmethod
    def encode(cls, obj: PIL.Image):
        if (
            isinstance(obj, PIL.Image.Image)
            and hasattr(obj, "filename")
            and ("." + obj.format) in cls.EXTENSIONS
        ):
            # avoid re-encoding, relevant for JPEGs
            orig = pathlib.Path(obj.filename)
            with orig.open("rb") as src:
                data = src.read()
            return data

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().detach()
        if isinstance(obj, np.ndarray):
            obj = PIL.Image.fromarray(obj)
        if not isinstance(obj, PIL.Image.Image):
            raise TypeError(
                "Can only serialize PIL.Image|np.ndarray|torch.Tensor objects"
            )
        mem = io.BytesIO()
        obj.save(mem, format=cls.FORMAT)
        return mem.getvalue()

    @classmethod
    def load(cls, fp) -> PIL.Image:
        fp = cls.check_fp(fp)
        return PIL.Image.open(fp)


class JPGFormat(ImageFormat):

    EXTENSIONS = [".jpg", ".jpeg"]
    FORMAT = "jpeg"


class PNGFormat(ImageFormat):

    EXTENSIONS = [".png"]
    FORMAT = "png"


class AudioFormat(CompressedFileFormat):
    @classmethod
    def save(cls, obj: Tuple[np.ndarray, int], fp):
        fp = cls.check_fp(fp)
        sf.write(fp, obj, samplerate=44_100)

    @classmethod
    def load(cls, fp) -> Tuple[np.ndarray, int]:
        fp = cls.check_fp(fp)
        return sf.read(fp)


class WAVFormat(AudioFormat):
    EXTENSIONS = [".wav"]


class FLACFormat(AudioFormat):
    EXTENSIONS = [".flac"]


class OGGFormat(AudioFormat):
    EXTENSIONS = [".ogg"]


class MsgpackFormat(FileFormat):

    EXTENSIONS = [".msgpack"]

    @classmethod
    def encode(cls, data: Any) -> bytes:
        return msgpack.packb(data)

    @classmethod
    def decode(cls, data: bytes) -> Any:
        return msgpack.unpackb(data, strict_map_key=False) #, raw=True)


class PlaintextFormat(FileFormat):

    EXTENSIONS = [".txt", ".log"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        if isinstance(obj, list):
            obj = "\n".join(obj)
        with fp.open("w") as f:
            f.write(obj)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return f.read().strip().split("\n")

