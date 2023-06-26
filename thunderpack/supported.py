from typing import Dict, List

from .compression import *
from .compression import ChainedFileCompression, CompressionFormat
from .formats import *
from .formats import FileFormat


class InvalidExtensionError(Exception):
    def __init__(self, extension: str):
        message = f"Unsupported extension '.{extension}'"
        super().__init__(message)
        self.extension = extension


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
        extension = extension.strip(".").lower()
        if extension not in cls._formats and "." in extension:
            file_ext, compression_ext = extension.split(".")
            compressed_fmt = ChainedFileCompression(
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
        return cls._formats.copy()

    @classmethod
    def compression_extensions(cls) -> List[str]:
        return [
            ext
            for ext, fileformat in cls._formats.items()
            if issubclass(fileformat, CompressionFormat)
        ]

    @classmethod
    def compression_formats(cls) -> List[FileFormat]:
        return {
            ext: fileformat
            for ext, fileformat in cls._formats.items()
            if issubclass(fileformat, CompressionFormat)
        }


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
    BinaryFormat,
    OGGFormat,
    WAVFormat,
    FLACFormat,
    FeatherFormat,
    GzipCompression,
    LZ4Compression,
    ZstdCompression,
    SnappyCompression,
    BZ2Compression,
):
    SupportedFormats.register(format_cls)
