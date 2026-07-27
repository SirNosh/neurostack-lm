from src.stage1r.audit import expert_flop_audit


def test_dense_expert_execution_counts_all_four_experts():
    audit = expert_flop_audit(batch_size=2, tokens=16, cycles=2)
    assert audit["computed_experts"] == 4
    assert audit["selected_experts"] == 2
    assert audit["actual_expert_flops"] == (
        2 * audit["hypothetical_selected_only_flops"]
    )
