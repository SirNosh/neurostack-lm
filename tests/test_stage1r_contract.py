import json
from pathlib import Path


def test_stage1r_run_matrix_has_seven_systems_three_seeds():
    config = json.loads(Path("configs/stage1r.json").read_text(encoding="utf-8"))
    systems = {"R0", "R1", "R2", "R3", "R3+aux", "R4", "R5"}
    matrix = config["run_matrix"]
    assert len(matrix) == 21
    assert {row["system"] for row in matrix} == systems
    assert {row["seed"] for row in matrix} == {1729, 2718, 3141}
    assert len({(row["system"], row["seed"]) for row in matrix}) == 21


def test_stage1r_capacity_audit_workload_is_frozen():
    config = json.loads(Path("configs/stage1r.json").read_text(encoding="utf-8"))
    assert config["capacity_audit_workload"] == {
        "batch_size": 1,
        "sequence_length": 512,
        "passes": 3,
        "episodic_entries": 128,
        "retrieval_breadth": 4,
    }
