import pytest
from unittest.mock import patch
import torch
from spotpython.utils.device import getDevice

def test_get_device_auto_cpu():
    with patch('torch.cuda.is_available') as mock_cuda, patch('torch.backends.mps.is_available') as mock_mps:
        mock_cuda.return_value = False
        mock_mps.return_value = False
        assert getDevice() == 'cpu'

def test_get_device_auto_cuda():
    with patch('torch.cuda.is_available') as mock_cuda, patch('torch.backends.mps.is_available') as mock_mps:
        mock_cuda.return_value = True
        mock_mps.return_value = False
        assert getDevice() == 'cuda:0'

def test_get_device_auto_mps():
    with patch('torch.cuda.is_available') as mock_cuda, patch('torch.backends.mps.is_available') as mock_mps:
        mock_cuda.return_value = False
        mock_mps.return_value = True
        assert getDevice() == 'mps'

def test_get_device_explicit_cpu():
    assert getDevice('cpu') == 'cpu'

def test_get_device_explicit_cuda_available():
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = True
        assert getDevice('cuda:0') == 'cuda:0'

def test_get_device_explicit_cuda_unavailable():
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = False
        with pytest.raises(ValueError, match="CUDA device requested but no CUDA device is available."):
            getDevice('cuda:0')

def test_get_device_explicit_mps_available():
    with patch('torch.backends.mps.is_available') as mock_mps:
        mock_mps.return_value = True
        assert getDevice('mps') == 'mps'

def test_get_device_explicit_mps_unavailable():
    with patch('torch.backends.mps.is_available') as mock_mps:
        mock_mps.return_value = False
        with pytest.raises(ValueError, match="MPS device requested but MPS is not available."):
            getDevice('mps')

def test_get_device_invalid():
    with pytest.raises(ValueError, match="Unrecognized device: invalid_device. Valid options are 'cpu', 'cuda:x', or 'mps'."):
        getDevice('invalid_device')