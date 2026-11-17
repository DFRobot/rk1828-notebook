# -*- coding: utf-8 -*-
"""RKNN3 Runtime (librknn3_api.so) 的 ctypes 封装。

面向 notebook 的最小可用接口：

    from rknn3 import Rknn3Model, find_devices

    model = Rknn3Model("xxx.rknn", "xxx.weight")
    outputs = model.infer([input_np])   # -> list[np.ndarray]
    model.close()

调用链（与官方 C demo 一致）：
    rknn3_init -> rknn3_load_model_from_path -> rknn3_model_init
    -> rknn3_query(输入输出属性) -> rknn3_create_mem / rknn3_mem_sync
    -> rknn3_run -> rknn3_destroy_mem / rknn3_destroy
"""
import ctypes
import os

import numpy as np

_LIB_PATH = "/usr/lib/librknn3_api.so"

RKNN3_MAX_DIMS = 16
RKNN3_MAX_STRIDE_DIMS = RKNN3_MAX_DIMS + 1
RKNN3_MAX_NAME_LEN = 256
RKNN3_MAX_DEVS = 64
RKNN3_MAX_DEV_LEN = 64
RKNN3_MAX_NPU_NODE_NUM = 128

# 查询命令
QUERY_IN_OUT_NUM = 0
QUERY_INPUT_ATTR = 1
QUERY_OUTPUT_ATTR = 2
QUERY_SDK_VERSION = 5

# mem_sync 方向
SYNC_TO_DEVICE = 0x1
SYNC_FROM_DEVICE = 0x2

# rknn3_tensor_type -> numpy dtype
_DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.uint8,
    4: np.int16,
    5: np.uint16,
    6: np.int32,
    7: np.uint32,
    8: np.int64,
    9: np.uint64,
    10: np.bool_,
    13: np.uint16,  # BFLOAT16 按 2 字节读出，调用方自行解释
}
_DTYPE_NAME = {0: "FP32", 1: "FP16", 2: "INT8", 3: "UINT8", 4: "INT16",
               5: "UINT16", 6: "INT32", 7: "UINT32", 8: "INT64", 9: "UINT64",
               10: "BOOL", 13: "BFLOAT16"}
_LAYOUT_NAME = {0: "UNDEFINED", 1: "NCHW", 2: "NHWC", 3: "NC1HWC2"}


class _QuantInfo(ctypes.Structure):
    _fields_ = [("scale", ctypes.c_float), ("zero_point", ctypes.c_int32)]


class TensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("name", ctypes.c_char * RKNN3_MAX_NAME_LEN),
        ("n_dims", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * RKNN3_MAX_DIMS),
        ("aligned_size", ctypes.c_uint64),
        ("n_stride", ctypes.c_uint32),
        ("stride", ctypes.c_uint64 * RKNN3_MAX_STRIDE_DIMS),
        ("n_elems", ctypes.c_uint32),
        ("dtype", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("qnt_info", _QuantInfo),
        ("n_orig_dims", ctypes.c_uint32),
        ("orig_shape", ctypes.c_uint32 * RKNN3_MAX_DIMS),
        ("orig_layout", ctypes.c_int),
        ("core_id", ctypes.c_int),
    ]

    @property
    def np_dtype(self):
        return _DTYPE_MAP.get(self.dtype, np.float32)

    @property
    def shape_tuple(self):
        return tuple(self.shape[i] for i in range(self.n_dims))

    @property
    def name_str(self):
        return self.name.decode(errors="replace")

    def __repr__(self):
        return (f"<TensorAttr {self.name_str} shape={self.shape_tuple} "
                f"dtype={_DTYPE_NAME.get(self.dtype, self.dtype)} "
                f"layout={_LAYOUT_NAME.get(self.layout, self.layout)} "
                f"scale={self.qnt_info.scale:.6g} zp={self.qnt_info.zero_point}>")


class TensorMem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("buffer_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("core_id", ctypes.c_int32),
        ("priv_data", ctypes.c_void_p),
        ("mem_type", ctypes.c_int),
    ]


class Tensor(ctypes.Structure):
    _fields_ = [("mem", ctypes.POINTER(TensorMem)),
                ("attr", ctypes.POINTER(TensorAttr))]


class _Config(ctypes.Structure):
    _fields_ = [
        ("priority", ctypes.c_int32),
        ("run_timeout", ctypes.c_uint32),
        ("run_core_mask", ctypes.c_uint32),
        ("user_mem_weight", ctypes.c_uint8),
        ("user_mem_internal", ctypes.c_uint8),
        ("user_sram", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 128),
    ]


class _IONum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class _NodeMemInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("free", ctypes.c_uint64)]


class _DevMemInfo(ctypes.Structure):
    _fields_ = [("node_num", ctypes.c_uint32),
                ("sys_total", ctypes.c_uint64),
                ("sys_free", ctypes.c_uint64),
                ("node_mem_info", _NodeMemInfo * RKNN3_MAX_NPU_NODE_NUM)]


class _Device(ctypes.Structure):
    _fields_ = [("id", ctypes.c_char * RKNN3_MAX_DEV_LEN),
                ("type", ctypes.c_char * RKNN3_MAX_DEV_LEN),
                ("mem_info", _DevMemInfo)]


class _Devices(ctypes.Structure):
    _fields_ = [("n_devices", ctypes.c_uint32),
                ("devices", _Device * RKNN3_MAX_DEVS)]


class _SdkVersion(ctypes.Structure):
    _fields_ = [("api_version", ctypes.c_char * 256),
                ("drv_version", ctypes.c_char * 256)]


_lib = None


def get_lib():
    """加载 /usr/lib/librknn3_api.so（单例）。"""
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise RuntimeError(
                f"{_LIB_PATH} 不存在，请先安装 RKNN3 运行时（rknn3-rk182x DEB 包）")
        _lib = ctypes.CDLL(_LIB_PATH)
        c = ctypes
        _lib.rknn3_init.argtypes = [c.POINTER(c.c_uint64), c.c_void_p]
        _lib.rknn3_load_model_from_path.argtypes = [c.c_uint64, c.c_char_p, c.c_char_p]
        _lib.rknn3_model_init.argtypes = [c.c_uint64, c.POINTER(_Config)]
        _lib.rknn3_destroy.argtypes = [c.c_uint64]
        _lib.rknn3_query.argtypes = [c.c_uint64, c.c_int, c.c_void_p, c.c_uint64]
        _lib.rknn3_run.argtypes = [c.c_uint64, c.POINTER(Tensor), c.c_uint32,
                                   c.POINTER(Tensor), c.c_uint32]
        _lib.rknn3_create_mem.argtypes = [c.c_uint64, c.c_uint64, c.c_int32, c.c_int]
        _lib.rknn3_create_mem.restype = c.POINTER(TensorMem)
        _lib.rknn3_destroy_mem.argtypes = [c.c_uint64, c.POINTER(TensorMem)]
        _lib.rknn3_mem_sync.argtypes = [c.c_uint64, c.POINTER(TensorMem), c.c_int]
        _lib.rknn3_find_devices.argtypes = [c.POINTER(_Devices)]
    return _lib


def find_devices():
    """枚举 RK1828 设备，返回 [(device_id, device_type), ...]。"""
    lib = get_lib()
    devs = _Devices()
    ret = lib.rknn3_find_devices(ctypes.byref(devs))
    if ret != 0:
        raise RuntimeError(f"rknn3_find_devices 失败: {ret}")
    return [(devs.devices[i].id.decode(), devs.devices[i].type.decode())
            for i in range(devs.n_devices)]


class Rknn3Model:
    """一个加载到 RK1828 上的 RKNN3 模型（权重分离格式 .rknn + .weight）。"""

    def __init__(self, model_path, weight_path=None, core_mask=0x1):
        if weight_path is None:
            weight_path = os.path.splitext(model_path)[0] + ".weight"
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)

        lib = get_lib()
        self._lib = lib
        self._ctx = ctypes.c_uint64(0)
        ret = lib.rknn3_init(ctypes.byref(self._ctx), None)
        if ret != 0:
            raise RuntimeError(f"rknn3_init 失败: {ret}")
        try:
            ret = lib.rknn3_load_model_from_path(
                self._ctx, model_path.encode(), weight_path.encode())
            if ret != 0:
                raise RuntimeError(f"rknn3_load_model_from_path 失败: {ret}")
            cfg = _Config()
            cfg.run_core_mask = core_mask
            ret = lib.rknn3_model_init(self._ctx, ctypes.byref(cfg))
            if ret != 0:
                raise RuntimeError(f"rknn3_model_init 失败: {ret}")
        except Exception:
            lib.rknn3_destroy(self._ctx)
            self._ctx = None
            raise

        io_num = _IONum()
        self._query(QUERY_IN_OUT_NUM, io_num)
        self.input_attrs = self._query_attrs(QUERY_INPUT_ATTR, io_num.n_input)
        self.output_attrs = self._query_attrs(QUERY_OUTPUT_ATTR, io_num.n_output)

        # 为每个输入/输出分配设备内存
        self._in_mems = [self._alloc(a) for a in self.input_attrs]
        self._out_mems = [self._alloc(a) for a in self.output_attrs]
        self._inputs = (Tensor * len(self.input_attrs))()
        self._outputs = (Tensor * len(self.output_attrs))()
        self._attr_bufs = []  # 防止 attr 被 GC
        for i, (t, m, a) in enumerate(zip(self._inputs, self._in_mems, self.input_attrs)):
            pa = ctypes.pointer(a)
            self._attr_bufs.append(pa)
            t.mem = m
            t.attr = pa
        for i, (t, m, a) in enumerate(zip(self._outputs, self._out_mems, self.output_attrs)):
            pa = ctypes.pointer(a)
            self._attr_bufs.append(pa)
            t.mem = m
            t.attr = pa

    def _query(self, cmd, obj):
        ret = self._lib.rknn3_query(self._ctx, cmd, ctypes.byref(obj),
                                    ctypes.sizeof(obj))
        if ret != 0:
            raise RuntimeError(f"rknn3_query(cmd={cmd}) 失败: {ret}")

    def _query_attrs(self, cmd, n):
        attrs = []
        for i in range(n):
            a = TensorAttr()
            a.index = i
            self._query(cmd, a)
            attrs.append(a)
        return attrs

    def _alloc(self, attr):
        size = max(attr.aligned_size,
                   attr.n_elems * np.dtype(attr.np_dtype).itemsize, 1)
        mem = self._lib.rknn3_create_mem(self._ctx, size, attr.core_id, 1)
        if not mem:
            raise RuntimeError(f"rknn3_create_mem({size}) 失败")
        return mem

    def infer(self, inputs):
        """inputs: list[np.ndarray]（按 input_attrs 顺序），返回 list[np.ndarray]。"""
        if len(inputs) != len(self.input_attrs):
            raise ValueError(f"需要 {len(self.input_attrs)} 个输入，给了 {len(inputs)} 个")
        for arr, attr, mem in zip(inputs, self.input_attrs, self._in_mems):
            data = np.ascontiguousarray(arr, dtype=attr.np_dtype)
            buf = data.tobytes()
            if len(buf) > mem.contents.size:
                raise ValueError(
                    f"输入 {attr.name_str} 需要 {len(buf)}B，超过分配的 {mem.contents.size}B")
            ctypes.memmove(mem.contents.virt_addr, buf, len(buf))
            ret = self._lib.rknn3_mem_sync(self._ctx, mem, SYNC_TO_DEVICE)
            if ret != 0:
                raise RuntimeError(f"mem_sync(to_device) 失败: {ret}")

        ret = self._lib.rknn3_run(self._ctx, self._inputs, len(self._inputs),
                                  self._outputs, len(self._outputs))
        if ret != 0:
            raise RuntimeError(f"rknn3_run 失败: {ret}")

        outs = []
        for attr, mem in zip(self.output_attrs, self._out_mems):
            ret = self._lib.rknn3_mem_sync(self._ctx, mem, SYNC_FROM_DEVICE)
            if ret != 0:
                raise RuntimeError(f"mem_sync(from_device) 失败: {ret}")
            n = attr.n_elems
            arr = np.frombuffer(
                (ctypes.c_char * (n * np.dtype(attr.np_dtype).itemsize)).from_address(
                    mem.contents.virt_addr),
                dtype=attr.np_dtype, count=n).copy()
            shape = attr.shape_tuple
            if shape and int(np.prod(shape)) == n:
                arr = arr.reshape(shape)
            outs.append(arr)
        return outs

    def sdk_version(self):
        v = _SdkVersion()
        self._query(QUERY_SDK_VERSION, v)
        return v.api_version.decode(), v.drv_version.decode()

    def close(self):
        if getattr(self, "_ctx", None):
            for mem in self._in_mems + self._out_mems:
                self._lib.rknn3_destroy_mem(self._ctx, mem)
            self._lib.rknn3_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
