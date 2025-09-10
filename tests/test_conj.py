import torch
import pytest

from model.tools import conj


def test_conj_on_native_complex_tensor():
    z = torch.tensor([1 + 2j, -3 - 4j], dtype=torch.cfloat)
    result = conj(z)
    assert result.is_complex(), "Result should retain complex dtype"
    expected = torch.conj(z)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conj_on_valid_real_representation(dtype):
    real = torch.tensor([1.0, -2.0], dtype=dtype)
    imag = torch.tensor([3.0, -4.0], dtype=dtype)
    stacked = torch.stack([real, imag], dim=-1)  # shape (..., 2)
    assert stacked.stride(-1) == 1, "Last dim should be contiguous"
    result = conj(stacked)
    # Should now be complex type
    assert result.is_complex()
    expected = torch.view_as_complex(stacked).conj()
    assert torch.allclose(result, expected)


def test_conj_with_float16_raises():
    real = torch.randn(2, dtype=torch.float16)
    imag = torch.randn(2, dtype=torch.float16)
    stacked = torch.stack([real, imag], dim=-1)
    with pytest.raises(TypeError):
        conj(stacked)


def test_conj_with_invalid_shape_raises():
    tensor = torch.randn(3, 3, dtype=torch.float32)
    with pytest.raises(ValueError):
        conj(tensor)


def test_conj_with_noncontiguous_stride_raises():
    # Create valid dtype and size but make last dim non contiguous
    x = torch.randn(2, 4, dtype=torch.float32)
    # Reshape to have last dimension 2, but via a transpose to make noncontiguous
    complex_like = x.view(2, 2, 2).transpose(1, 2)  # now stride of last dim not 1
    assert complex_like.size(-1) == 2
    with pytest.raises(ValueError):
        conj(complex_like)


# Optionally, test ValueError message content consistency
def test_error_message_contains_dtype_and_shape():
    tensor = torch.ones(5, dtype=torch.int32)
    with pytest.raises(ValueError) as exc:
        conj(tensor)
    msg = str(exc.value)
    assert "Unsupported tensor type or shape" in msg
    assert "dtype=torch.int32" in msg and "shape=torch.Size" in msg
