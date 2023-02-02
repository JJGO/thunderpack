import bz2
import gzip
import itertools
from typing import Literal

import brotli
import lz4.frame
import snappy
import zstd

from .formats import CompressedFileFormat, FileFormat


class CompressionFormat(FileFormat):
    pass


class GzipCompression(CompressionFormat):

    EXTENSIONS = [".gz"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return gzip.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return gzip.decompress(data)


class LZ4Compression(CompressionFormat):

    EXTENSIONS = [".lz4"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return lz4.frame.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class ZstdCompression(CompressionFormat):

    EXTENSIONS = [".zst"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return zstd.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return zstd.decompress(data)


class BZ2Compression(CompressionFormat):

    EXTENSIONS = [".bz2"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return bz2.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return bz2.decompress(data)


class SnappyCompression(CompressionFormat):

    EXTENSIONS = [".snappy"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return snappy.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return snappy.decompress(data)


class BrotliCompression(CompressionFormat):

    EXTENSIONS = [".br"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return brotli.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return brotli.decompress(data)


Compression = Literal["lz4", "zstd", "gzip", "bz2", "snappy"]

_compression_codecs = {
    "lz4": LZ4Compression,
    "zstd": ZstdCompression,
    "brotli": BrotliCompression,
    "snappy": SnappyCompression,
    "gzip": GzipCompression,
}


def autocompress(data: bytes, compression: Compression) -> bytes:
    codec = _compression_codecs[compression]
    return codec.encode(data)


def autodecompress(data: bytes, compression: Compression) -> bytes:
    codec = _compression_codecs[compression]
    return codec.decode(data)


class ChainedFileCompression(CompressedFileFormat):
    def __init__(self, file_format: FileFormat, compression_format: CompressionFormat):
        if issubclass(file_format, CompressedFileFormat):
            raise TypeError(f"file_format {file_format} supports native compression")
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
