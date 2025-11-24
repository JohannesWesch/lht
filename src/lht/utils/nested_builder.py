"""
Utility to build GeometricCoordinates from nested list document structure.

Format:
    document = [
        [section1_sent1, section1_sent2, ...],
        [section2_sent1, ...],
        ...
    ]
"""

from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer

from lht.core.attention import GeometricCoordinates


def build_coords_from_nested_list(
    document: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    device: torch.device = None,
) -> Tuple[List[int], GeometricCoordinates]:
    """
    Convert nested document list into flat input_ids and GeometricCoordinates.

    Hierarchy:
    - Level 0: Tokens
    - Level 1: Sentences
    - Level 2: Sections

    Args:
        document: Nested list of strings [[sent, sent], [sent], ...]
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (truncates if exceeded)
        device: torch device

    Returns:
        (input_ids, coords)
    """
    device = device or torch.device("cpu")

    flat_input_ids = []
    token_parents = []  # Level 0 -> 1 (Token -> Sentence)
    sentence_parents = []  # Level 1 -> 2 (Sentence -> Section)

    # We also need to track sections -> Root (Level 2 -> 3, but here root is just identity)
    # Or just treat sections as top level. The user requested 3 levels (token, sent, sec).
    # So:
    # Tokens point to Sentence ID
    # Sentences point to Section ID
    # Sections point to their own ID (top level)

    current_sentence_id = 0
    current_section_id = 0

    # Iterate through sections
    for section in document:
        # Each section is a list of sentences
        for sentence in section:
            # Tokenize sentence
            # Add CLS/SEP only if needed, or relying on special tokens in text?
            # Assuming standard BERT-like tokenization: [CLS] sent [SEP]
            # But for long docs, usually we just concat tokens.
            # Let's assume simple concatenation for now, maybe adding a separator if needed.
            # Ideally, we tokenize without special tokens first, then manage them.

            tokens = tokenizer.encode(sentence, add_special_tokens=False)

            # Check length limit
            if len(flat_input_ids) + len(tokens) > max_length:
                remaining = max_length - len(flat_input_ids)
                tokens = tokens[:remaining]
                # If we truncated, we stop processing after this
                stop_processing = True
            else:
                stop_processing = False

            if not tokens:
                continue

            flat_input_ids.extend(tokens)

            # Assign parent (sentence ID) for each token
            token_parents.extend([current_sentence_id] * len(tokens))

            # Assign parent (section ID) for this sentence
            sentence_parents.append(current_section_id)

            current_sentence_id += 1

            if stop_processing:
                break

        current_section_id += 1
        if len(flat_input_ids) >= max_length:
            break

    # Build coords - we only need coordinates for actual tokens, not hierarchy nodes
    # Each token gets (level, logical_time) describing its place in hierarchy
    # We don't create separate coordinate entries for sentences/sections

    # For tokens: level=0, logical_time=sentence_id
    levels = [0] * len(flat_input_ids)
    logical_times = token_parents  # Each token's logical_time is its parent sentence ID

    coords = GeometricCoordinates(
        levels=torch.tensor(levels, dtype=torch.long, device=device),
        logical_times=torch.tensor(logical_times, dtype=torch.long, device=device),
        physical_positions=torch.arange(
            len(flat_input_ids), dtype=torch.long, device=device
        ),
    )

    return flat_input_ids, coords
