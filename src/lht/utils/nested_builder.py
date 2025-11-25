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
    add_doc_token: bool = True,
) -> Tuple[List[int], HierarchicalPositions]:
    """
    Convert nested document list into flat input_ids and HierarchicalPositions.

    Builds sparse enumeration vectors using cumsum:
    - Level 0: All tokens (dense enumeration: 1, 2, 3, ...)
    - Level 1: Sentence boundaries only (sparse: 0, 0, 0, 1, 0, 2, ...)
    - Level 2: Section boundaries only (sparse: 0, 0, 0, 0, 0, 1, ...)

    Hierarchy:
    - Level 0: Tokens
    - Level 1: Sentences (marked at FIRST token of new sentence)
    - Level 2: Sections (marked at FIRST token of new section)

    Special [DOC] token (position 0):
    - L0=1 (participates in local token attention)
    - L1=0 (doesn't participate in sentence-level)
    - L2=1 (participates in section-level - aggregates document structure)

    Args:
        document: Nested list of strings [[sent, sent], [sent], ...]
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (truncates if exceeded)
        device: torch device
        add_doc_token: If True, prepend [CLS] token as document-level aggregator

    Returns:
        (input_ids, positions)
    """
    device = device or torch.device("cpu")

    flat_input_ids = []
    is_sentence_boundary = []  # 1 at first token of new sentence, 0 elsewhere
    is_section_boundary = []  # 1 at first token of new section, 0 elsewhere

    # Add [DOC] token at the beginning (uses [CLS] token from tokenizer)
    if add_doc_token:
        doc_token_id = tokenizer.cls_token_id
        if doc_token_id is None:
            # Fallback if tokenizer doesn't have [CLS]
            doc_token_id = (
                tokenizer.bos_token_id
                if tokenizer.bos_token_id
                else tokenizer.eos_token_id
            )

        flat_input_ids.append(doc_token_id)
        is_sentence_boundary.append(0)  # [DOC] doesn't participate in sentence-level
        is_section_boundary.append(
            0
        )  # [DOC] doesn't participate in cumsum (will be set later)

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

            # Mark boundaries for this sentence
            is_first_sent_in_section = sent_idx == 0

            for i in range(len(tokens)):
                flat_input_ids.append(tokens[i])

                # Mark first token of each sentence (including the very first one)
                is_first_token_of_sentence = i == 0
                is_sentence_boundary.append(1 if is_first_token_of_sentence else 0)

                # Mark first token of each section (including the very first one)
                is_first_token_of_section = i == 0 and is_first_sent_in_section
                is_section_boundary.append(1 if is_first_token_of_section else 0)

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

    # Special handling for [DOC] token
    if add_doc_token and N > 0:
        # Set [DOC] token's L2 to 1 (first section enum) for section-level attention
        # This allows [DOC] to attend to all sections without consuming a section number
        level_2[0] = 1

    positions = HierarchicalPositions(
        [
            level_0.to(device),
            level_1.to(device),
            level_2.to(device),
        ]
    )

    return flat_input_ids, positions
