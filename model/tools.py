import sys, os, random, json, uuid, time, argparse, logging, logging.config
import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from random import randint
from collections import defaultdict as ddict, Counter
from ordered_set import OrderedSet
from pprint import pprint

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import Parameter as Param
from torch.utils.data import DataLoader
from torch_scatter import scatter_add


np.set_printoptions(precision=4)


def set_gpu(gpus) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    config_dict = json.load(open(config_dir + "log_config.json"))
    config_dict["handlers"]["file_handler"]["filename"] = log_dir + name.replace(
        ":", "-"
    )
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = "%(asctime)s - [%(levelname)s] - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results["count"])

    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round((left_results["mr"] + right_results["mr"]) / (2 * count), 5)
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )

    for k in range(10):
        results["left_hits@{}".format(k + 1)] = round(
            left_results["hits@{}".format(k + 1)] / count, 5
        )
        results["right_hits@{}".format(k + 1)] = round(
            right_results["hits@{}".format(k + 1)] / count, 5
        )
        results["hits@{}".format(k + 1)] = round(
            (
                left_results["hits@{}".format(k + 1)]
                + right_results["hits@{}".format(k + 1)]
            )
            / (2 * count),
            5,
        )
    return results


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results["count"])

    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round((left_results["mr"] + right_results["mr"]) / (2 * count), 5)
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )

    for k in range(10):
        results["left_hits@{}".format(k + 1)] = round(
            left_results["hits@{}".format(k + 1)] / count, 5
        )
        results["right_hits@{}".format(k + 1)] = round(
            right_results["hits@{}".format(k + 1)] / count, 5
        )
        results["hits@{}".format(k + 1)] = round(
            (
                left_results["hits@{}".format(k + 1)]
                + right_results["hits@{}".format(k + 1)]
            )
            / (2 * count),
            5,
        )
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def com_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # if a.dim() >= 2 and a.size(-1) == 2 and b.dim() >= 2 and b.size(-1) == 2:
    # r1, i1 = a[..., 0], a[..., 1]
    # r2, i2 = b[..., 0], b[..., 1]
    # return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)
    if a.is_complex() and b.is_complex():
        return a * b

    # Legacy: real tensors shaped [..., 2]
    if a.dim() >= 1 and a.size(-1) == 2 and b.dim() >= 1 and b.size(-1) == 2:
        a_c = torch.view_as_complex(a)
        b_c = torch.view_as_complex(b)
        return a_c * b_c

    raise ValueError(
        f"Unsupported tensor format for com_mult(): "
        f"dtype/shapes: a dtype {a.dtype}, shape {a.shape}; b dtype {b.dtype}, shape {b.shape}"
    )


def conj(a):
    # if a.dim() >= 2 and a.size(-1) == 2:
    # a[..., 1] = -a[..., 1]
    # return a
    if a.is_complex():
        return torch.conj(a)
    # If it's real tensor with last dim size 2:
    if a.dtype in (torch.float32, torch.float64, torch.float16) and a.size(-1) == 2:
        a = torch.view_as_complex(a)
        return torch.conj(a)
    raise ValueError(
        f"Unsupported tensor type or shape for conj(): dtype={a.dtype}, shape={a.shape}"
    )


def cconv(a, b):
    return torch.fft.irfft(
        com_mult(torch.fft.rfft(a, 1), torch.fft.rfft(b, 1)), 1, n=a.shape[-1], dim=1
    )


def ccorr(a, b):
    return torch.fft.irfft(
        com_mult(conj(torch.fft.rfft(a, dim=1)), torch.fft.rfft(b, dim=1)),
        n=a.shape[-1],
        dim=1,
    )
