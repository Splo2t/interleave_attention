from __future__ import annotations


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def choices_block_any(choices: list[str]) -> str:
    labels = letters(len(choices))
    return "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
