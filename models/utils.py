from typing import Union, List
import gc
import torch


def str2torch_dtype(dtype: str, default=torch.float32) -> torch.dtype:
    """字符串形式转成torch.dtype形式"""
    if dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp32":
        return torch.float32
    return default


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]],
                         dtype=torch.float32) -> list:
    """获取model中所有可训练参数"""
    if not isinstance(model, list):
        model = [model]
    training_params = []
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)
                training_params.append(param)
    return training_params


def flush_vram():
    """清除显存缓存"""
    torch.cuda.empty_cache()
    gc.collect()


def quantization(model):
    """模型量化"""
    try:
        from optimum.quanto import freeze, qfloat8, quantize, QTensor
    except ModuleNotFoundError:
        raise ModuleNotFoundError("需要安装optimum库，请执行'pip install optimum-quanto'")
    quantization_type = qfloat8
    print("Quantizing transformer")
    quantize(model, weights=quantization_type)
    freeze(model)
    flush_vram()