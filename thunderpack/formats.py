from abc import ABC
import io
import itertools
import json
import gzip
import pathlib
import pickle
from typing import Any, Literal, Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
import yaml
import PIL.Image
import PIL.JpegImagePlugin
import lz4.frame
import zstd
import soundfile as sf

import pyarrow as pa
import pyarrow.parquet as pq

import msgpack
import msgpack_numpy as m

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


class CompressionFormat(FileFormat):
    pass


class BinaryFormat(FileFormat):

    EXTENSIONS = [".bin", ".BIN"]

    @classmethod
    def encode(cls, obj) -> bytes:
        return obj

    @classmethod
    def decode(cls, data: bytes) -> object:
        return data


class NpyFormat(FileFormat):

    EXTENSIONS = [".npy", ".NPY"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.save(fp, obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return np.load(fp)


class NpzFormat(FileFormat):

    EXTENSIONS = [".npz", ".NPZ"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.savez(fp, **obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return dict(np.load(fp))


class PtFormat(FileFormat):

    EXTENSIONS = [".pt", ".PT"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        torch.save(obj, fp)

    @classmethod
    def load(cls, fp):
        fp = cls.check_fp(fp)
        return torch.load(fp)


class YamlFormat(FileFormat):

    EXTENSIONS = [".yaml", ".YAML", ".yml", ".YML"]

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

    EXTENSIONS = [".json", ".JSON"]

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

    EXTENSIONS = [".jsonl", ".JSONL"]

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

    EXTENSIONS = [".csv", ".CSV"]

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


class ParquetFormat(FileFormat):

    # Tweaked version of write/read_parquet to support
    # storing .attrs metadata in the parquet metadata fields

    EXTENSIONS = [".parquet", ".PARQUET", ".pq", ".PQ"]

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


class PickleFormat(FileFormat):

    EXTENSIONS = [".pkl", ".PKL", ".pickle", ".PICKLE"]

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


class BaseImageFormat(FileFormat):
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


class JPGFormat(BaseImageFormat):

    EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG"]
    FORMAT = "jpeg"


class PNGFormat(BaseImageFormat):

    EXTENSIONS = [".png", ".PNG"]
    FORMAT = "png"


class BaseAudioFormat(FileFormat):
    @classmethod
    def save(cls, obj: Tuple[np.ndarray, int], fp):
        fp = cls.check_fp(fp)
        sf.write(fp, obj)

    @classmethod
    def load(cls, fp) -> Tuple[np.ndarray, int]:
        fp = cls.check_fp(fp)
        return sf.read(fp)


class WAVFormat(BaseAudioFormat):
    EXTENSIONS = [".wav", ".WAV"]


class FLACFormat(BaseAudioFormat):
    EXTENSIONS = [".flac", ".FLAC"]


class OGGFormat(BaseAudioFormat):
    EXTENSIONS = [".ogg", ".OGG"]


class GzipFormat(CompressionFormat):

    EXTENSIONS = [".gz", ".GZ"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return gzip.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return gzip.decompress(data)


class LZ4Format(CompressionFormat):

    EXTENSIONS = [".lz4", ".LZ4"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return lz4.frame.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class ZstdFormat(CompressionFormat):

    EXTENSIONS = [".zst", ".ZST"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return zstd.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return zstd.decompress(data)


class MsgpackFormat(FileFormat):

    EXTENSIONS = [".msgpack", ".MSGPACK"]

    @classmethod
    def encode(cls, data: Any) -> bytes:
        return msgpack.packb(data)

    @classmethod
    def decode(cls, data: bytes) -> Any:
        return msgpack.unpackb(data)


class PlaintextFormat(FileFormat):

    EXTENSIONS = [".txt", ".TXT", ".log", ".LOG"]

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


class InvalidExtensionError(Exception):
    def __init__(self, extension: str):
        message = f"Unsupported extension '.{extension}'"
        super().__init__(message)
        self.extension = extension


class CompressedFileFormat(FileFormat):
    def __init__(self, file_format: FileFormat, compression_format: CompressionFormat):
        if not issubclass(file_format, FileFormat):
            raise TypeError("file_format must be a FileFormat")
        if not issubclass(compression_format, CompressionFormat):
            raise TypeError("compression_format must be a CompressionFormat")
        self.file_format = file_format
        self.compression_format = compression_format
        self.EXTENSIONS = [
            fe + ce
            for fe, ce in itertools.product(
                self.file_format.EXTENSIONS, self.compression_format.EXTENSIONS
            )
        ]

    def __repr__(self) -> str:
        ff = f"{self.file_format.__module__}.{self.file_format.__name__}"
        cf = f"{self.compression_format.__module__}.{self.compression_format.__name__}"
        return f"{self.__class__.__module__}.{self.__class__.__name__}({ff}, {cf})"

    def save(self, obj, fp):
        data = self.file_format.encode(obj)
        self.compression_format.save(data, fp)

    def load(self, fp) -> object:
        data = self.compression_format.load(fp)
        return self.file_format.decode(data)

    def encode(self, obj) -> bytes:
        data = self.file_format.encode(obj)
        return self.compression_format.encode(data)

    def decode(self, data: bytes) -> object:
        data = self.compression_format.decode(data)
        return self.file_format.decode(data)


class SupportedFormats:

    _formats: Dict[str, FileFormat] = {}

    @classmethod
    def register(cls, format_cls: FileFormat, force=False):
        for extension in format_cls.EXTENSIONS:
            extension = extension.strip(".")
            assert force or extension not in cls._formats
            cls._formats[extension] = format_cls

    @classmethod
    def get_format(cls, extension: str) -> FileFormat:
        extension = extension.strip(".")
        if extension not in cls._formats and "." in extension:
            file_ext, compression_ext = extension.split(".")
            compressed_fmt = CompressedFileFormat(
                cls.get_format(file_ext), cls.get_format(compression_ext)
            )
            cls.register(compressed_fmt)

        fmt = cls._formats.get(extension, None)
        if fmt is not None:
            return fmt
        raise InvalidExtensionError(extension)

    @classmethod
    def extensions(cls) -> List[str]:
        return list(cls._formats.keys())

    @classmethod
    def formats(cls) -> List[FileFormat]:
        return list(set(cls._formats.values()))

    @classmethod
    def compression_extensions(cls) -> List[str]:
        return [e for e, f in cls._formats.items() if isinstance(f, CompressionFormat)]

    @classmethod
    def compression_formats(cls) -> List[FileFormat]:
        return list(
            set([f for f in cls.formats.values() if isinstance(f, CompressionFormat)])
        )


for format_cls in (
    NpyFormat,
    NpzFormat,
    PtFormat,
    YamlFormat,
    JsonFormat,
    JsonlFormat,
    CsvFormat,
    ParquetFormat,
    PickleFormat,
    PNGFormat,
    JPGFormat,
    MsgpackFormat,
    PlaintextFormat,
    GzipFormat,
    LZ4Format,
    ZstdFormat,
    BinaryFormat,
    OGGFormat,
    WAVFormat,
    FLACFormat,
):
    SupportedFormats.register(format_cls)
