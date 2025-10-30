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
    # Tokenize with consistent settings (no special tokens to match slot config)
    tokens = tokenizer(
        prompt,
        padding=padding,
        truncation=truncation,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,  # Must match determine_slot_config()
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
        # Debug: Show what tokens we actually got
        import sys
        print(
            f"\nDEBUG: Slot search failed\n"
            f"  Slot string:     '{slot_config.slot_string}'\n"
            f"  Expected T_slot: {T_slot}\n"
            f"  Prompt:          '{prompt}'\n"
            f"  Token IDs:       {input_ids.tolist()[:50]}... (first 50)\n"
            f"  add_special_tokens: False\n",
            file=sys.stderr,
        )
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


def determine_slot_config_in_context(
    tokenizer,
    slot_string: str,
    base_prompt: str,
    target_length: Optional[int] = None,
) -> SlotConfig:
    """
    Determine slot token sequence by analyzing tokenization in actual prompt context.

    This solves the boundary-token problem: tokenizers like BPE may produce different
    token IDs for a slot string depending on context (e.g., neighboring whitespace,
    punctuation). By tokenizing the actual base_prompt with/without the slot, we
    extract the exact token sequence as it appears in practice.

    Args:
        tokenizer: HuggingFace tokenizer
        slot_string: The slot placeholder string (e.g., "~ID~" or "<ID_token>")
        base_prompt: The actual prompt template containing the slot
                     (e.g., "a portrait photo of ~ID~, studio lighting")
        target_length: Optional expected token length for validation

    Returns:
        SlotConfig with token IDs as they appear in base_prompt context

    Raises:
        ValueError: If slot_string not found in base_prompt
        ValueError: If extracted length != target_length (when specified)
        ValueError: If token extraction fails (unable to locate insertion point)

    Example:
        >>> slot_config = determine_slot_config_in_context(
        ...     tokenizer,
        ...     slot_string="<ID_token>",
        ...     base_prompt="a portrait photo of <ID_token>, studio lighting",
        ...     target_length=4
        ... )
        >>> slot_config.T_slot  # Token IDs as they appear in the prompt
        [3968, 915, 6458, 54497]
    """
    # Validate slot in prompt
    if slot_string not in base_prompt:
        raise ValueError(
            f"Slot string '{slot_string}' not found in base_prompt '{base_prompt}'"
        )

    # Tokenize prompt without slot (replace with empty string)
    prompt_without = base_prompt.replace(slot_string, "")
    tokens_without = tokenizer(
        prompt_without,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids_without = tokens_without["input_ids"][0]

    # Tokenize prompt with slot
    tokens_with = tokenizer(
        base_prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids_with = tokens_with["input_ids"][0]

    # Remove padding for cleaner comparison
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        ids_without = ids_without[ids_without != pad_token_id]
        ids_with = ids_with[ids_with != pad_token_id]

    # Find longest common prefix
    prefix_len = 0
    max_prefix = min(len(ids_without), len(ids_with))
    while prefix_len < max_prefix and ids_with[prefix_len] == ids_without[prefix_len]:
        prefix_len += 1

    # Find longest common suffix
    suffix_len = 0
    max_suffix = min(len(ids_without), len(ids_with)) - prefix_len
    while suffix_len < max_suffix and ids_with[-(suffix_len + 1)] == ids_without[-(suffix_len + 1)]:
        suffix_len += 1

    # Extract slot token sequence
    slot_start = prefix_len
    slot_end = len(ids_with) - suffix_len

    if slot_start >= slot_end:
        raise ValueError(
            f"Failed to extract slot tokens from prompt.\n"
            f"  Slot string: '{slot_string}'\n"
            f"  Base prompt: '{base_prompt}'\n"
            f"  Tokens without slot: {ids_without.tolist()}\n"
            f"  Tokens with slot:    {ids_with.tolist()}\n"
            f"  Prefix length: {prefix_len}, Suffix length: {suffix_len}"
        )

    T_slot = ids_with[slot_start:slot_end].tolist()
    L = len(T_slot)

    # Validate target length if specified
    if target_length is not None and L != target_length:
        raise ValueError(
            f"Slot '{slot_string}' tokenizes to {L} tokens in context, "
            f"but target_length={target_length} was specified.\n"
            f"  Base prompt: '{base_prompt}'\n"
            f"  Token IDs:   {T_slot}\n"
            f"  Hint: Either adjust your slot string or change target_length."
        )

    return SlotConfig(slot_string=slot_string, T_slot=T_slot, L=L)
