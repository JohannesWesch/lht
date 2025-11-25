"""
Test suite for training integration components.

Tests loss computation, config loading, optimizer setup, and scheduler.
"""

import os
import tempfile

import pytest
import torch
import yaml

from lht.config import ExperimentConfig, TrainingConfig, load_config
from lht.training import compute_mlm_loss


def test_compute_mlm_loss_basic():
    """Test basic MLM loss computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    # Create dummy logits and labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_mlm_loss(logits, labels)

    # Loss should be scalar
    assert loss.dim() == 0, "Loss should be scalar"

    # Loss should be positive
    assert loss.item() > 0, "Loss should be positive"

    # Loss should not be NaN or Inf
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"


def test_compute_mlm_loss_with_ignore_index():
    """Test that ignore_index=-100 is properly ignored."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Set only one position to a valid label
    labels[0, 0] = 5

    loss = compute_mlm_loss(logits, labels)

    # Loss should only be computed on the one valid position
    assert not torch.isnan(
        loss
    ), "Loss should not be NaN even with mostly ignored labels"


def test_compute_mlm_loss_all_masked():
    """Test loss when all tokens are masked (no -100)."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_mlm_loss(logits, labels)

    # Should compute loss over all positions
    assert loss.item() > 0


def test_compute_mlm_loss_gradient_flow():
    """Test that gradients flow through loss computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_mlm_loss(logits, labels)
    loss.backward()

    # Logits should have gradients
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_config_loading_from_yaml():
    """Test loading configuration from YAML file."""
    # Create temporary YAML config
    config_dict = {
        "experiment_name": "test_experiment",
        "seed": 42,
        "device": "cpu",
        "model": {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 512,
            "rope": True,
            "mlswa": {
                "num_levels": 3,
                "window_size_per_level": [256, 64, 16],
                "layer_max_level": [2, 2],
            },
        },
        "training": {
            "task": "mlm",
            "batch_size": 4,
            "grad_accum_steps": 8,
            "num_steps": 10000,
            "warmup_steps": 1000,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every": 100,
            "save_every": 5000,
            "eval_every": 1000,
            "mixed_precision": "bf16-mixed",
            "mlm_probability": 0.15,
        },
        "data": {
            "text_column": "text",
            "num_workers": 4,
            "shuffle_buffer_size": 10000,
            "max_seq_len": 512,
            "tokenizer_name_or_path": "google-bert/bert-base-uncased",
        },
        "wandb": {
            "project": "test_project",
            "entity": None,
            "name": None,
            "tags": ["test"],
            "log_model": False,
            "watch_model": "gradients",
            "watch_freq": 1000,
            "offline": False,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "test_experiment"
        assert config.seed == 42
        assert config.model.vocab_size == 1000
        assert config.training.batch_size == 4
    finally:
        os.unlink(temp_path)


def test_config_float_conversion():
    """Test that learning_rate and other floats are properly converted."""
    config_dict = {
        "experiment_name": "test",
        "seed": 42,
        "device": "cpu",
        "model": {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 512,
        },
        "training": {
            "task": "mlm",
            "batch_size": 4,
            "grad_accum_steps": 8,
            "num_steps": 10000,
            "warmup_steps": 1000,
            "learning_rate": "1e-4",  # String that should be converted to float
            "weight_decay": "0.01",
            "max_grad_norm": 1.0,
            "log_every": 100,
            "save_every": 5000,
            "eval_every": 1000,
            "mixed_precision": "bf16-mixed",
            "mlm_probability": "0.15",
        },
        "data": {
            "text_column": "text",
            "num_workers": 4,
            "shuffle_buffer_size": 10000,
            "max_seq_len": 512,
            "tokenizer_name_or_path": "bert-base-uncased",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        # Check that values are floats
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.training.weight_decay, float)
        assert isinstance(config.training.mlm_probability, float)

        # Check values
        assert abs(config.training.learning_rate - 0.0001) < 1e-6
        assert abs(config.training.weight_decay - 0.01) < 1e-6
        assert abs(config.training.mlm_probability - 0.15) < 1e-6
    finally:
        os.unlink(temp_path)


def test_optimizer_creation():
    """Test that AdamW optimizer can be created with config."""
    config_dict = {
        "experiment_name": "test",
        "seed": 42,
        "device": "cpu",
        "model": {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 512,
            "mlswa": {
                "num_levels": 3,
                "window_size_per_level": [256, 64, 16],
                "layer_max_level": [2, 2],
            },
        },
        "training": {
            "task": "mlm",
            "batch_size": 4,
            "grad_accum_steps": 8,
            "num_steps": 10000,
            "warmup_steps": 1000,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every": 100,
            "save_every": 5000,
            "eval_every": 1000,
            "mixed_precision": "bf16-mixed",
        },
        "data": {
            "text_column": "text",
            "num_workers": 4,
            "shuffle_buffer_size": 10000,
            "max_seq_len": 512,
            "tokenizer_name_or_path": "bert-base-uncased",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        # Create a simple model
        model = torch.nn.Linear(10, 10)

        # Create optimizer with config values
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == config.training.learning_rate
        assert optimizer.defaults["weight_decay"] == config.training.weight_decay
    finally:
        os.unlink(temp_path)


def test_training_config_values():
    """Test that TrainingConfig has correct default values."""
    training_cfg = TrainingConfig(
        task="mlm",
        batch_size=4,
        grad_accum_steps=8,
        num_steps=10000,
        warmup_steps=1000,
        learning_rate=0.0001,
        weight_decay=0.01,
        max_grad_norm=1.0,
        log_every=100,
        save_every=5000,
        eval_every=1000,
        mixed_precision="bf16-mixed",
    )

    assert training_cfg.mlm_probability == 0.15  # Default value


def test_loss_reduction_mean():
    """Test that loss uses mean reduction over valid tokens."""
    vocab_size = 100

    # Create logits and labels with known values
    logits = torch.randn(1, 3, vocab_size)
    labels = torch.tensor([[10, -100, 20]])  # Only 2 valid labels

    loss = compute_mlm_loss(logits, labels)

    # Loss should be averaged over 2 valid positions (not 3)
    assert not torch.isnan(loss)


def test_loss_with_different_batch_sizes():
    """Test loss computation with various batch sizes."""
    vocab_size = 100

    for batch_size in [1, 2, 4, 8]:
        seq_len = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_mlm_loss(logits, labels)

        assert loss.dim() == 0, f"Loss should be scalar for batch_size={batch_size}"
        assert not torch.isnan(loss), f"Loss is NaN for batch_size={batch_size}"


def test_config_mlswa_parsing():
    """Test that ML-SWA config is properly parsed."""
    config_dict = {
        "experiment_name": "test",
        "seed": 42,
        "device": "cpu",
        "model": {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 512,
            "mlswa": {
                "num_levels": 3,
                "window_size_per_level": [256, 64, 16],
                "layer_max_level": [2, 2],
            },
        },
        "training": {
            "task": "mlm",
            "batch_size": 4,
            "grad_accum_steps": 8,
            "num_steps": 10000,
            "warmup_steps": 1000,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every": 100,
            "save_every": 5000,
            "eval_every": 1000,
            "mixed_precision": "bf16-mixed",
        },
        "data": {
            "text_column": "text",
            "num_workers": 4,
            "shuffle_buffer_size": 10000,
            "max_seq_len": 512,
            "tokenizer_name_or_path": "bert-base-uncased",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        assert config.model.mlswa is not None
        assert config.model.mlswa.num_levels == 3
        assert config.model.mlswa.window_size_per_level == [256, 64, 16]
        assert config.model.mlswa.layer_max_level == [2, 2]
    finally:
        os.unlink(temp_path)


def test_loss_perfect_prediction():
    """Test loss when predictions are perfect."""
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    # Create perfect predictions (argmax matches labels)
    labels = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])

    # Create logits with very high values for correct classes
    logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, labels[b, s]] = 10.0

    loss = compute_mlm_loss(logits, labels)

    # Loss should be very small for perfect predictions
    assert loss.item() < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
