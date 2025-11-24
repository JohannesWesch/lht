# LHT Test Suite

Comprehensive unit tests for the Learned Hierarchical Transformer (LHT) training setup.

## Test Overview

**Total Tests**: 90 tests across 6 test modules
- **Passing**: 59 tests (CPU-compatible)
- **Skipped**: 31 tests (require CUDA for FlexAttention)

## Test Modules

### 1. `test_nested_builder.py` (11 tests)
Tests the nested document to coordinates pipeline.

**Coverage**:
- `build_coords_from_nested_list()` with various nested structures
- Tokenization of nested documents
- Coordinate assignment for tokens/sentences/sections
- Max length truncation behavior
- Empty section/sentence handling
- Device placement (CPU/CUDA)

**Key Tests**:
- Single/multi-section documents
- Coordinate consistency (one coord per token)
- Proper sentence ID assignment via `logical_time`

### 2. `test_data_collator.py` (14 tests)
Tests the `HierarchicalDataCollator` for batch preparation.

**Coverage**:
- Batch padding with varying sequence lengths
- Coordinate tensor padding/truncation to match input_ids
- MLM masking (15% of tokens)
- Batch stacking with GeometricCoordinates
- Integration with BERT tokenizer

**Key Tests**:
- MLM mask token usage and percentage validation
- Label handling (ignore_index=-100 for unmasked tokens)
- Coordinate-input_ids length matching

### 3. `test_geometric_attention.py` (20 tests)
Tests geometric attention mechanism and Manhattan distance.

**Coverage**:
- GeometricCoordinates initialization and device transfer
- Manhattan distance calculations (siblings, parent-child, cross-section)
- FlexAttention integration (with GPU)
- **Mock-based tests** for CPU (5 tests)

**Key Tests**:
- Distance=0 for siblings (same parent, same level)
- Distance=1 for parent-child pairs
- Tensor transposition ([B,N,H,D] ↔ [B,H,N,D])
- Parameter passing to create_block_mask (mocked)

**Mock Tests** (run on CPU):
- `test_create_geometric_mask_calls_create_block_mask_with_correct_params`
- `test_geometric_attention_transposes_tensors_correctly`
- `test_geometric_mask_fn_logic`
- `test_geometric_attention_with_window_mocked`
- `test_geometric_mask_respects_max_level`

### 4. `test_sliding_window.py` (14 tests, all require CUDA)
Tests per-level sliding window constraints.

**Coverage**:
- Per-level window sizes [256, 64, 16]
- Same-level window enforcement
- Cross-level attention (window doesn't apply)
- Combined Manhattan + window constraints
- Edge cases (boundaries, zero window, very large windows)

**Note**: All tests skipped on CPU-only systems (require FlexAttention).

### 5. `test_model_architecture.py` (19 tests)
Tests LHTEncoder and GeometricTransformerBlock architecture.

**Coverage**:
- Model initialization with correct dimensions
- Weight initialization (mean=0, std=0.02)
- Final LayerNorm existence
- No weight tying between embeddings and MLM head
- Forward pass with coordinates (requires CUDA)
- Parameter count (~53.5M for HDT-E config)

**Key Tests**:
- Initialization tests (CPU-compatible)
- Forward pass tests (require CUDA, 8 skipped)
- Per-layer max_level assignment
- Window sizes assignment

### 6. `test_training_integration.py` (12 tests)
Tests training components and configuration.

**Coverage**:
- Loss computation (CrossEntropyLoss with ignore_index=-100)
- Masked token loss only
- Config loading (float conversion for learning_rate, etc.)
- Optimizer initialization (AdamW)
- Geometry config parsing

**Key Tests**:
- MLM loss with various masking scenarios
- Config YAML loading and type conversion
- Gradient flow through loss computation

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Module
```bash
pytest tests/test_nested_builder.py -v
```

### Run Only CPU-Compatible Tests
```bash
pytest tests/ -v -k "not cuda"
```

### Run Only Mock Tests
```bash
pytest tests/test_geometric_attention.py -k "mock" -v
```

## Test Requirements

- **pytest**: Test framework
- **torch**: PyTorch (CPU tests work without CUDA)
- **transformers**: HuggingFace transformers
- **unittest.mock**: Python standard library (for mocking)

## CUDA Requirements

Tests that require CUDA are marked with `@requires_cuda` or module-level `pytestmark`:
- FlexAttention-based tests (geometric attention, sliding window)
- Model forward pass tests (use geometric attention)

These tests are **automatically skipped** on CPU-only systems.

## Key Implementation Details Tested

### 1. Coordinate Structure
- Coordinates contain **only token-level entries** (no separate hierarchy nodes)
- Each token has: `level=0`, `logical_time=sentence_id`
- Length: `len(coords.levels) == len(input_ids)`

### 2. Manhattan Distance
- **Siblings** (same sentence): distance = 0
- **Parent-child**: distance = 1 (|Δlevel|=1, |Δtime|=0)
- **Cross-section**: distance > 1 (different logical_time)

### 3. Tensor Shapes
- Input to model: `[B, N, H, D]` (batch, seq, heads, head_dim)
- FlexAttention uses: `[B, H, N, D]` (transposed)
- Output transposed back: `[B, N, H, D]`

### 4. MLM Masking
- 15% of tokens masked
- Labels: -100 for unmasked, original token_id for masked
- Loss computed only on masked positions

### 5. Data Collation
- Coordinates padded/truncated to match `max_len` of batch
- All batch elements padded to same length
- Coordinates stacked into `GeometricCoordinates` object

## Future Improvements

1. **Add integration tests** for full training loop
2. **Add GPU tests** when CUDA is available
3. **Add visualization tests** for attention patterns
4. **Add benchmarking tests** for performance
5. **Add tests for RoPE** (rotary position embeddings)

## Contributing

When adding new features, please:
1. Add corresponding unit tests
2. Use mocking for GPU-dependent code to enable CPU testing
3. Mark GPU-required tests with `@requires_cuda`
4. Ensure tests are deterministic (use `torch.manual_seed()`)
