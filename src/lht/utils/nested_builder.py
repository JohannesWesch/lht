"""
Utility to build HierarchicalPositions from nested list document structure.

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

from lht.core.attention import HierarchicalPositions


def build_coords_from_nested_list(
    document: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    device: torch.device = None,
) -> Tuple[List[int], HierarchicalPositions]:
    """
    Convert nested document list into flat input_ids and HierarchicalPositions.

    Builds sparse enumeration vectors using cumsum:
    - Level 0: All tokens (dense enumeration: 1, 2, 3, ...)
    - Level 1: Sentence boundaries only (sparse: 1, 0, 0, 2, 0, 3, ...)
    - Level 2: Section boundaries only (sparse: 1, 0, 0, 0, 0, 2, ...)

    Hierarchy:
    - Level 0: Tokens
    - Level 1: Sentences (marked at sentence end)
    - Level 2: Sections (marked at section end)

    Args:
        document: Nested list of strings [[sent, sent], [sent], ...]
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (truncates if exceeded)
        device: torch device

    Returns:
        (input_ids, positions)
    """
    device = device or torch.device("cpu")

    flat_input_ids = []
    is_sentence_boundary = []  # 1 at sentence end, 0 elsewhere
    is_section_boundary = []  # 1 at section end, 0 elsewhere

    # Iterate through sections
    for section_idx, section in enumerate(document):
        # Each section is a list of sentences
        for sent_idx, sentence in enumerate(section):
            tokens = tokenizer.encode(sentence, add_special_tokens=False)

            # Check length limit
            if len(flat_input_ids) + len(tokens) > max_length:
                remaining = max_length - len(flat_input_ids)
                tokens = tokens[:remaining]
                stop_processing = True
            else:
                stop_processing = False

            if not tokens:
                continue

            flat_input_ids.extend(tokens)

            # Mark boundaries
            is_last_sent_in_sec = sent_idx == len(section) - 1
            for i in range(len(tokens)):
                is_last_token_in_sent = i == len(tokens) - 1

                is_sentence_boundary.append(1 if is_last_token_in_sent else 0)
                is_section_boundary.append(
                    1 if (is_last_token_in_sent and is_last_sent_in_sec) else 0
                )

            if stop_processing:
                break

        if len(flat_input_ids) >= max_length:
            break

    N = len(flat_input_ids)

    if N == 0:
        # Empty document - return empty positions
        positions = HierarchicalPositions(
            [
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
            ]
        )
        return flat_input_ids, positions

    # Build enumeration vectors using cumsum
    # Level 0: All tokens participate (dense enumeration)
    level_0 = torch.arange(1, N + 1, dtype=torch.long)

    # Level 1: Sentence boundaries only
    sent_mask = torch.tensor(is_sentence_boundary, dtype=torch.long)
    level_1 = torch.cumsum(sent_mask, dim=0) * sent_mask  # Zero out non-boundaries

    # Level 2: Section boundaries only
    sec_mask = torch.tensor(is_section_boundary, dtype=torch.long)
    level_2 = torch.cumsum(sec_mask, dim=0) * sec_mask  # Zero out non-boundaries

    positions = HierarchicalPositions(
        [
            level_0.to(device),
            level_1.to(device),
            level_2.to(device),
        ]
    )

    return flat_input_ids, positions
