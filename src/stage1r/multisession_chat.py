from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import Stage1RExample


def adapt_msc_file(
    path: Path, *, split: str, conversation_limit: int = 100, seed: int = 1729
) -> list[Stage1RExample]:
    frame = pd.read_parquet(path)
    conversation_ids = sorted(
        frame["dialoug_id"].unique(),
        key=lambda value: __import__("hashlib").sha256(
            f"{seed}:{split}:{value}".encode()
        ).digest(),
    )[:conversation_limit]
    output: list[Stage1RExample] = []
    for dialogue_id in conversation_ids:
        rows = frame[frame["dialoug_id"] == dialogue_id].sort_values("session_id")
        timestamp = 0
        for row in rows.itertuples(index=False):
            history: list[str] = []
            for turn, (speaker, utterance) in enumerate(zip(row.speaker, row.dialogue)):
                first = timestamp == 0
                prompt = "\n".join(history[-8:])
                prompt += ("\n" if prompt else "") + f"{speaker}:"
                output.append(
                    Stage1RExample(
                        example_id=(
                            f"msc:{split}:{dialogue_id}:session-{row.session_id}:turn-{turn}"
                        ),
                        family="multisession_chat",
                        input_text=prompt,
                        target_text=str(utterance),
                        session_id=f"msc:{split}:{dialogue_id}",
                        task_context="multisession_chat",
                        timestamp=timestamp,
                        support_spans=[],
                        support_item_ids=[],
                        retrieval_target_ids=[],
                        encode_target=True,
                        verifier_label=None,
                        relation_label=None,
                        boundary_label=None,
                        reset_pfc=first,
                        reset_working_memory=first,
                        reset_fast_weights=first,
                        reset_episodic_memory=first,
                    )
                )
                history.append(f"{speaker}: {utterance}")
                timestamp += 1
    return output


def msc_stage1r_splits(
    root: Path, *, conversation_limit: int = 100, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    files = {
        split: next((root / "data").glob(f"{split}-*.parquet"))
        for split in ("train", "validation", "test")
    }
    return {
        "train": adapt_msc_file(
            files["train"], split="train", conversation_limit=conversation_limit, seed=seed
        ),
        "dev": adapt_msc_file(
            files["validation"], split="dev", conversation_limit=conversation_limit, seed=seed
        ),
        "test": adapt_msc_file(
            files["test"], split="test", conversation_limit=conversation_limit, seed=seed
        ),
    }, list(files.values())
