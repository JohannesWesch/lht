"""
Convenience coordinate builders for common cases.

These are convenience wrappers around core.build_coords()
for common 2-level and 3-level hierarchies.
"""

import torch

from ..core import GeometricCoordinates, build_coords


def build_two_level_coords(
    num_tokens: int,
    sentence_boundaries: list[int],
    device: torch.device = None,
) -> GeometricCoordinates:
    """
    Convenience builder for tokens + sentences.

    Args:
        num_tokens: number of base tokens
        sentence_boundaries: token indices where sentences end
        device: torch device

    Returns:
        GeometricCoordinates

    Example:
        build_two_level_coords(6, [2, 5])
        → 6 tokens + 2 sentences
    """
    num_sentences = len(sentence_boundaries)

    # Build parent maps
    token_parents = []
    sent_id = 0
    for i in range(num_tokens):
        token_parents.append(sent_id)
        if sent_id < len(sentence_boundaries) and i == sentence_boundaries[sent_id]:
            sent_id += 1

    sentence_parents = list(range(num_sentences))  # top level

    return build_coords([token_parents, sentence_parents], device=device)


def build_three_level_coords(
    num_tokens: int,
    sentence_boundaries: list[int],
    section_boundaries: list[int],
    device: torch.device = None,
) -> GeometricCoordinates:
    """
    Convenience builder for tokens + sentences + sections.

    Args:
        num_tokens: number of base tokens
        sentence_boundaries: token indices where sentences end
        section_boundaries: sentence indices where sections end
        device: torch device

    Returns:
        GeometricCoordinates

    Example:
        build_three_level_coords(9, [2, 5, 8], [1])
        → 9 tokens + 3 sentences + 2 sections
    """
    num_sentences = len(sentence_boundaries)
    num_sections = len(section_boundaries) + 1

    # Token parents (which sentence)
    token_parents = []
    sent_id = 0
    for i in range(num_tokens):
        token_parents.append(sent_id)
        if sent_id < len(sentence_boundaries) and i == sentence_boundaries[sent_id]:
            sent_id += 1

    # Sentence parents (which section)
    sentence_parents = []
    sec_id = 0
    for sent_id in range(num_sentences):
        sentence_parents.append(sec_id)
        if sec_id < len(section_boundaries) and sent_id == section_boundaries[sec_id]:
            sec_id += 1

    # Section parents (top level)
    section_parents = list(range(num_sections))

    return build_coords(
        [token_parents, sentence_parents, section_parents], device=device
    )


def build_flat_coords(
    seq_len: int,
    device: torch.device = None,
) -> GeometricCoordinates:
    """
    Convenience builder for flat sequences (no hierarchy).

    All tokens at level=0, sequential logical_time.
    Gives standard sliding window attention.

    Args:
        seq_len: sequence length
        device: torch device

    Returns:
        GeometricCoordinates with flat structure
    """
    device = device or torch.device("cpu")

    return GeometricCoordinates(
        levels=torch.zeros(seq_len, dtype=torch.long, device=device),
        logical_times=torch.arange(seq_len, dtype=torch.long, device=device),
        physical_positions=torch.arange(seq_len, dtype=torch.long, device=device),
    )
