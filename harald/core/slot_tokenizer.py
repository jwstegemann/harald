"""Slot tokenization and position finding utilities."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch


@dataclass
class SlotConfig:
    """
    Configuration for slot-based token injection.

    Attributes:
        slot_string: The placeholder string (e.g., "~ID~")
        T_slot: Token ID sequence for the slot
        L: Fixed length of the slot in tokens
        s: Start position of slot in base prompt (for teacher training)
    """

    slot_string: str
    T_slot: List[int]
    L: int
    s: Optional[int] = None  # Set during base prompt tokenization


def tokenize_and_find_slot(
    tokenizer,
    prompt: str,
    slot_config: SlotConfig,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Tokenize a prompt and find all slot positions.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt containing one or more slot strings
        slot_config: SlotConfig with slot token sequence T_slot
        padding: Tokenizer padding mode (default: "max_length")
        truncation: Whether to truncate (default: True)
        return_tensors: Format for output (default: "pt")

    Returns:
        Tuple of (input_ids tensor, list of slot start positions)

    Raises:
        ValueError: If slot not found in tokenized prompt
    """
    # Tokenize with consistent settings
    tokens = tokenizer(
        prompt,
        padding=padding,
        truncation=truncation,
        max_length=tokenizer.model_max_length,
        add_special_tokens=True,  # Include [CLS], [SEP], etc.
        return_tensors=return_tensors,
    )

    input_ids = tokens["input_ids"][0]  # [S]
    T_slot = slot_config.T_slot
    L = slot_config.L

    # Find all occurrences of T_slot as a subsequence
    positions = []
    i = 0
    while i <= len(input_ids) - L:
        # Check if T_slot matches at position i
        match = all(input_ids[i + j].item() == T_slot[j] for j in range(L))
        if match:
            positions.append(i)
            i += L  # Skip past this slot to avoid overlapping matches
        else:
            i += 1

    if not positions:
        raise ValueError(
            f"Slot '{slot_config.slot_string}' (T_slot={T_slot}) not found in prompt '{prompt}'. "
            f"Tokenizer version/config mismatch?"
        )

    return input_ids, positions


def find_all_slot_positions(
    tokenizer,
    prompt: str,
    slot_config: SlotConfig,
) -> List[Tuple[int, int]]:
    """
    Find all slot positions in a prompt.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        slot_config: SlotConfig with slot details

    Returns:
        List of (start, length) tuples for each slot occurrence (left-to-right order)

    Raises:
        ValueError: If slot not found
    """
    _, positions = tokenize_and_find_slot(tokenizer, prompt, slot_config)
    return [(s, slot_config.L) for s in positions]


def determine_slot_config(
    tokenizer,
    target_length: int = 4,
    candidate_strings: Optional[List[str]] = None,
) -> SlotConfig:
    """
    Determine a slot string that tokenizes to exactly target_length tokens.

    Args:
        tokenizer: HuggingFace tokenizer
        target_length: Desired token length (default: 4)
        candidate_strings: List of candidate slot strings to try
                          (default: ["~ID~", "~identity~", "~ID_token~", "~SLOT~"])

    Returns:
        SlotConfig for the first candidate that matches target_length

    Raises:
        ValueError: If no candidate produces target_length tokens
    """
    if candidate_strings is None:
        candidate_strings = ["~ID~", "~identity~", "~ID_token~", "~SLOT~"]

    for candidate in candidate_strings:
        # Tokenize without special tokens to get pure slot length
        tokens = tokenizer(
            candidate,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]

        # Count non-padding tokens
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            # If no pad token, all tokens are non-padding
            non_pad_ids = input_ids
        else:
            non_pad_ids = input_ids[input_ids != pad_token_id]

        L = len(non_pad_ids)

        if L == target_length:
            T_slot = non_pad_ids.tolist()
            return SlotConfig(slot_string=candidate, T_slot=T_slot, L=L)

    # No candidate matched
    raise ValueError(
        f"No slot string from {candidate_strings} tokenizes to exactly {target_length} tokens. "
        f"Tokenizer: {tokenizer.__class__.__name__}, vocab_size={tokenizer.vocab_size}"
    )
