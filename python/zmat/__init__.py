"""
zmat — A portable data compression/decompression library.

Re-exports the C backend (_zmat) and adds NumPy-aware helpers that
capture and restore array dtype, shape, and memory order alongside
compressed data, mirroring the ``[output, info] = zmat(input)`` /
``output = zmat(input, info)`` pattern from the MATLAB/Octave toolbox.

Basic API (bytes in, bytes out — no NumPy dependency):
    zmat.compress(data, method='zlib', level=1)
    zmat.decompress(data, method='zlib')
    zmat.encode(data, method='base64')
    zmat.decode(data, method='base64')
    zmat.zmat(data, iscompress=1, method='zlib', ...)   # low-level

NumPy-aware API:
    compressed, info = zmat.compress(arr, info=True)
    restored_arr     = zmat.decompress(compressed, info=info)
"""

from _zmat import compress as _compress
from _zmat import decode
from _zmat import decompress as _decompress
from _zmat import encode
from _zmat import zmat as _zmat_c

__all__ = ["compress", "decompress", "encode", "decode", "zmat"]


def compress(data, method="zlib", level=1, info=False):
    """Compress *data* using the requested algorithm.

    Parameters
    ----------
    data : bytes, bytearray, memoryview, or numpy.ndarray
        Input to compress.  Any object that supports the buffer protocol
        is accepted.  When *data* is a :class:`numpy.ndarray` **and**
        *info=True*, the array metadata (dtype, shape, memory order) is
        captured so that :func:`decompress` can reconstruct the original
        array exactly.
    method : str, optional
        Compression algorithm.  One of ``'zlib'`` (default), ``'gzip'``,
        ``'lzma'``, ``'lzip'``, ``'lz4'``, ``'lz4hc'``, ``'zstd'``,
        ``'base64'``, ``'blosc2blosclz'``, ``'blosc2lz4'``,
        ``'blosc2lz4hc'``, ``'blosc2zlib'``, ``'blosc2zstd'``.
    level : int, optional
        Compression level: ``1`` = library default, higher values give
        better compression at the cost of speed.
    info : bool, optional
        When *True* and *data* is a :class:`numpy.ndarray`, return a
        ``(compressed_bytes, info_dict)`` tuple instead of plain bytes.
        The *info_dict* mirrors the MATLAB ``info`` struct and contains:

        - ``'type'``   — NumPy dtype string (e.g. ``'float64'``)
        - ``'shape'``  — tuple of array dimensions
        - ``'byte'``   — bytes per element (``data.itemsize``)
        - ``'method'`` — the compression method used
        - ``'order'``  — ``'F'`` for Fortran-contiguous, ``'C'`` otherwise

        When *info=True* but *data* is not an ndarray, the tuple
        ``(compressed_bytes, None)`` is returned so callers can always
        unpack two values.

    Returns
    -------
    bytes
        Compressed data (when *info=False*).
    tuple[bytes, dict | None]
        ``(compressed_bytes, info_dict)`` when *info=True*.

    Examples
    --------
    Basic bytes round-trip::

        compressed = zmat.compress(b"hello " * 1000)
        original   = zmat.decompress(compressed)

    NumPy array round-trip::

        import numpy as np
        arr = np.random.rand(100, 100)
        compressed, info = zmat.compress(arr, info=True)
        restored = zmat.decompress(compressed, info=info)   # ndarray
        assert np.array_equal(restored, arr)
    """
    if info:
        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                order = "F" if np.isfortran(data) else "C"
                arr_info = {
                    "type": str(data.dtype),
                    "shape": tuple(data.shape),
                    "byte": data.itemsize,
                    "method": method,
                    "order": order,
                }
                # always serialise as C-contiguous bytes
                flat = np.ascontiguousarray(data).tobytes()
                compressed = _compress(flat, method=method, level=level)
                return compressed, arr_info
        except ImportError:
            pass

        # non-ndarray with info=True: compress normally, return (bytes, None)
        return _compress(data, method=method, level=level), None

    return _compress(data, method=method, level=level)


def decompress(data, method="zlib", info=None):
    """Decompress *data*.

    Parameters
    ----------
    data : bytes or bytearray
        Compressed input.
    method : str, optional
        Compression algorithm (default ``'zlib'``).  Ignored when *info*
        is provided — the method stored in ``info['method']`` is used
        instead, matching the MATLAB toolbox behaviour.
    info : dict or None, optional
        Info dict returned by :func:`compress` with ``info=True``.
        When provided, the raw decompressed bytes are reinterpreted as a
        :class:`numpy.ndarray` with the original dtype, shape, and memory
        order.  If NumPy is not installed the raw ``bytes`` are returned.

    Returns
    -------
    bytes
        Decompressed data (when *info* is ``None`` or NumPy is absent).
    numpy.ndarray
        Restored array with original dtype and shape (when *info* is given
        and NumPy is available).

    Examples
    --------
    ::

        import numpy as np
        arr = np.eye(50, dtype=np.float32)
        compressed, info = zmat.compress(arr, method='lz4', info=True)
        restored = zmat.decompress(compressed, info=info)
        assert restored.dtype == arr.dtype
        assert restored.shape == arr.shape
        assert np.array_equal(restored, arr)
    """
    if info is not None:
        actual_method = info.get("method", method)
        raw = _decompress(data, method=actual_method)
        try:
            import numpy as np

            dtype = np.dtype(info["type"])
            shape = tuple(info["shape"])
            order = info.get("order", "C")
            # frombuffer on immutable bytes is read-only; .copy() makes it writable
            arr = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
            if order == "F":
                arr = np.asfortranarray(arr)
            return arr
        except ImportError:
            return raw

    return _decompress(data, method=method)


def zmat(data, iscompress=1, method="zlib", nthread=1, shuffle=1, typesize=4):
    """Low-level compression/decompression interface with full parameter control.

    Parameters
    ----------
    data : bytes, bytearray, or buffer-protocol object
        Input data.
    iscompress : int
        ``1`` to compress at the default level (default), ``0`` to
        decompress, or a negative integer to set the compression level
        (e.g. ``-9`` for maximum compression).
    method : str
        Compression algorithm (default ``'zlib'``).
    nthread : int
        Thread count for blosc2 codecs (default ``1``).
    shuffle : int
        Byte-shuffle flag for blosc2: ``0`` = disabled, ``1`` = enabled
        (default ``1``).
    typesize : int
        Element byte size used by the blosc2 byte-shuffle filter
        (default ``4``).

    Returns
    -------
    bytes
        Compressed or decompressed data.
    """
    return _zmat_c(
        data,
        iscompress=iscompress,
        method=method,
        nthread=nthread,
        shuffle=shuffle,
        typesize=typesize,
    )
