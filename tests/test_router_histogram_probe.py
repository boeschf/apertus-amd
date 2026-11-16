"""Portable pinned-native AST/side-effect guard checks, NOT Torch/GPU correctness."""

import ast
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/investigations/performance-next/restart-20261113/routing-regression"

# Exact dedented source excerpts from the checksum-verified parent blob.
# Keep the readable native bodies here so a clean checkout needs no ignored
# artifacts or external repository. Both resulting byte strings must match the
# independent pinned extraction hashes in router_histogram_probe.py.
BASE_NATIVE = '''def quantile_balancing(
    self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
):
    """Apply the configured quantile-balancing routing estimator.

    The average methods gather TP/CP values and accumulate one quantile per
    microbatch for later averaging. ``legacy_average`` uses raw logits for
    compatibility with older QB checkpoints, while ``average`` uses bounded
    router scores. The histogram method performs no forward-pass communication:
    it accumulates local counts that are pooled once at the batch boundary.
    ``marin_histogram`` uses raw logits and a live global histogram per forward,
    then token-weights its quantiles across microbatches at the batch boundary.
    ``marin_exact`` uses raw logits too but retains every microbatch's margins and runs
    Marin's single global-batch estimator once at the step boundary (no forward collectives).

    Returns probs, routing_map and, when a capacity factor is set, the selection margin
    ``(qb_scores - qb_beta) - alpha`` used to rank tokens for dropping, where ``alpha`` is
    each token's Top-(k+1) biased score. It is the quantity whose per-expert quantile the
    beta update estimates, so dropping the smallest margins acts like raising the
    overloaded expert's threshold for this microbatch. It is None otherwise.
    """
    assert (
        not self.config.moe_router_fusion
    ), "Quantile balancing routing does not support moe_router_fusion."
    assert (
        self.config.moe_router_num_groups is None and self.config.moe_router_group_topk is None
    ), "Quantile balancing routing does not support group-limited routing."

    local_num_tokens = logits.shape[0]
    # The average estimators gather QB scores across TP/CP. The histogram estimator
    # does not use this group during the forward pass.
    gather_group = self.tp_cp_group
    gather_size = gather_group.size() if gather_group is not None else 1

    should_update_beta = self.training and torch.is_grad_enabled()

    with torch.no_grad():
        logits_fp32 = logits.detach().to(dtype=torch.float32)
        if self.score_function == "sigmoid":
            scores = torch.sigmoid(logits_fp32)
        elif self.score_function == "softmax":
            scores = torch.softmax(logits_fp32, dim=-1)
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        # NOTE: Legacy average is so that previous checkpoints that used the average method can still be loaded and work correctly.
        # This backwards compatibility should probably be removed in 1 or 2 months from this git blame
        qb_scores = (
            logits_fp32
            if self.config.moe_router_quantile_balancing_method in (
                'legacy_average', 'marin_histogram', 'marin_exact'
            )
            else scores
        )
        biased_scores = qb_scores - self.qb_beta
        use_histogram = (
            self.config.moe_router_quantile_balancing_method == 'histogram'
        )
        use_marin = self.config.moe_router_quantile_balancing_method == 'marin_histogram'
        use_exact = self.config.moe_router_quantile_balancing_method == 'marin_exact'
        compute_drop_priority = (
            self.config.moe_expert_capacity_factor is not None
            and self.config.moe_token_drop_policy == "probs"
        )
        if (
            should_update_beta and (use_histogram or use_marin or use_exact)
        ) or compute_drop_priority:
            topk_result = biased_scores.topk(self.topk + 1, dim=1)
            indices = topk_result.indices[:, : self.topk]
        else:
            indices = biased_scores.topk(self.topk, dim=1).indices
        drop_priority = None
        if compute_drop_priority:
            drop_priority = biased_scores - topk_result.values[:, -1:]

        if should_update_beta:
            if use_marin:
                if self.tp_dp_cp_group is None:
                    raise RuntimeError("marin_histogram requires a TP+DP+CP process group")
                beta_local, token_count = marin_qb_histogram_update(
                    qb_scores,
                    topk_result.values[:, -1],
                    self.qb_beta,
                    self.topk,
                    self.config.moe_router_quantile_balancing_marin_num_bins,
                    group=self.tp_dp_cp_group,
                    padding_mask=padding_mask,
                )
                self.qb_beta_accum.add_(beta_local * token_count)
                self.qb_beta_count.add_(token_count)
            elif use_exact:
                # No forward collectives: margins wait for the step boundary, where
                # Marin's estimator runs once over every token of the global batch.
                self._store_marin_exact_margins(
                    qb_scores - topk_result.values[:, -1:], padding_mask
                )
            elif use_histogram:
                # Hand the mask down instead of compacting the rows here:
                # scores[~padding_mask] lowers to nonzero(), which synchronizes the
                # device and makes the shape data-dependent, so the router can no
                # longer be CUDA-graph captured. Counts are identical.
                self.qb_histogram.add_(
                    compute_qb_histogram(
                        scores,
                        topk_result.values[:, -1],
                        self.qb_beta,
                        self.config.moe_router_quantile_balancing_num_bins,
                        padding_mask=padding_mask,
                    )
                )
            else:
                beta_scores = qb_scores
                beta_padding_mask = padding_mask
                if gather_size > 1:
                    # TODO: we could cache a reusable gather buffer sized for the max token count.
                    gathered_beta_scores = torch.empty(
                        (local_num_tokens * gather_size, self.config.num_moe_experts),
                        dtype=logits_fp32.dtype,
                        device=logits_fp32.device,
                    )
                    scores_work = torch.distributed.all_gather_into_tensor(
                        gathered_beta_scores,
                        beta_scores.contiguous(),
                        group=gather_group,
                        async_op=True,
                    )
                    mask_work = None
                    if padding_mask is not None:
                        local_padding_mask = padding_mask.to(dtype=torch.uint8)
                        beta_padding_mask = torch.empty(
                            local_num_tokens * gather_size,
                            dtype=local_padding_mask.dtype,
                            device=local_padding_mask.device,
                        )
                        mask_work = torch.distributed.all_gather_into_tensor(
                            beta_padding_mask,
                            local_padding_mask,
                            group=gather_group,
                            async_op=True,
                        )

                    scores_work.wait()
                    if mask_work is not None:
                        mask_work.wait()

                    beta_scores = gathered_beta_scores
                    if padding_mask is not None:
                        beta_padding_mask = beta_padding_mask.bool()

                if beta_padding_mask is not None:
                    beta_scores = beta_scores[~beta_padding_mask]
                if beta_scores.numel() > 0:
                    _, beta_local = qb_dual_update(
                        beta_scores, self.topk, self.qb_beta, update_beta=True
                    )
                    self.qb_beta_accum.add_(beta_local)
                    self.qb_beta_count.add_(1)

    # QB only picks the experts; reuse the shared score function for the probs.
    probs, routing_map = topk_routing_with_score_function(
        logits,
        self.topk,
        use_pre_softmax=self.config.moe_router_pre_softmax,
        scaling_factor=self.config.moe_router_topk_scaling_factor,
        score_function=self.score_function,
        fused=False,
        precomputed_indices=indices,
        sigmoid_normalization_epsilon=(
            1e-9
            if self.config.moe_router_quantile_balancing_method
            in ('marin_histogram', 'marin_exact')
            else 1e-20
        ),
    )
    return probs, routing_map, drop_priority

def _apply_expert_bias(
    self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
):
    """
    Update expert bias and tokens_per_expert
    Prevent extra local tokens accumulation on evaluation or activation recomputation
    """
    if torch.is_grad_enabled():
        with torch.no_grad():
            if padding_mask is not None:
                routing_map = routing_map & (~padding_mask.unsqueeze(-1))
            self.local_tokens_per_expert += routing_map.sum(dim=0)

collect_training_metrics = (
    self.training
    and torch.is_grad_enabled()
    and self.config.moe_router_violation_metrics
)

collect_inference_metrics = (
    not self.training
    and not torch.is_grad_enabled()
    and self.config.moe_router_inference_violation_metrics
)
'''

# This is the exact committed delta, not the test oracle. The resulting complete
# excerpt is checked against the independently captured fixed-blob digest.
FIX_NATIVE = BASE_NATIVE.replace('''        if (
            should_update_beta and (use_histogram or use_marin or use_exact)
        ) or compute_drop_priority:''', '''        # Checkpoint forward and replay must select the same experts at ties.
        # Only statistics accumulation below depends on grad mode.
        if (
            (self.training and use_histogram)
            or (should_update_beta and (use_marin or use_exact))
            or compute_drop_priority
        ):''', 1)


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "router_histogram_probe", ROOT / "experiments/validation/router_histogram_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_tree(probe, commit):
    raw = (BASE_NATIVE if commit == probe.BASE else FIX_NATIVE).encode()
    assert probe.digest(raw) == probe.EXCERPT_SHA256[commit]
    return ast.parse(raw.decode())


def test_portable_excerpts_match_retained_committed_provenance(probe):
    for commit in (probe.BASE, probe.FIX):
        native_tree(probe, commit)
    # Artifacts are optional in a clean checkout; when present they must agree
    # with the embedded portable fixtures and their independent pinned hashes.
    if not (EVIDENCE / "provenance.json").exists():
        return
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    assert provenance["parent"] == probe.BASE and provenance["commit"] == probe.FIX
    assert provenance["only_router_changed"] and provenance["only_selection_test_ast_changed"]
    assert provenance["path"] == probe.ROUTER
    for commit in (probe.BASE, probe.FIX):
        record = provenance["sources"][commit]
        assert record["router_sha256"] == probe.SOURCE_SHA256[commit]
        assert record["excerpt_sha256"] == probe.EXCERPT_SHA256[commit]
        assert len(record["ranges"]) == 4
        raw = (BASE_NATIVE if commit == probe.BASE else FIX_NATIVE).encode()
        assert (EVIDENCE / f"router-{commit[:12]}.py.txt").read_bytes() == raw


def test_clean_checkout_does_not_require_ignored_artifacts(probe, tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "EVIDENCE", tmp_path)
    test_portable_excerpts_match_retained_committed_provenance(probe)
    for commit in (probe.BASE, probe.FIX):
        assert probe.conditions(native_tree(probe, commit), training=True, grad_enabled=True,
                                method="histogram")["topk_request"] == 9


def test_every_retained_statement_except_selection_condition_is_unchanged(probe):
    base, fixed = (native_tree(probe, commit) for commit in (probe.BASE, probe.FIX))
    assert ast.dump(probe.selector(base).test) != ast.dump(probe.selector(fixed).test)
    probe.selector(base).test = deepcopy(probe.selector(fixed).test)
    assert ast.dump(base) == ast.dump(fixed)
    # Includes the complete statistics body, score-space selection, drop priority,
    # shared probability call, load-update body, and both metric conditions.


@pytest.mark.parametrize("training,grad_enabled,method,capacity,drop_policy", list(itertools.product(
    (False, True), (False, True),
    ("histogram", "marin_histogram", "marin_exact", "average", "legacy_average"),
    (None, 0, 1.25), ("probs", "position"))))
def test_actual_native_condition_matrix(probe, training, grad_enabled, method, capacity, drop_policy):
    kwargs = dict(training=training, grad_enabled=grad_enabled, method=method,
                  capacity=capacity, drop_policy=drop_policy)
    base, fixed = (probe.conditions(native_tree(probe, commit), **kwargs)
                   for commit in (probe.BASE, probe.FIX))
    drop_priority = capacity is not None and drop_policy == "probs"
    expected_base = 9 if (training and grad_enabled and method in (
        "histogram", "marin_histogram", "marin_exact")) or drop_priority else 8
    assert base["topk_request"] == expected_base
    affected = training and not grad_enabled and method == "histogram" and not drop_priority
    assert fixed["topk_request"] == (9 if affected else expected_base)
    for result in (base, fixed):
        assert result["statistics_update"] == (training and grad_enabled)
        # The old native expert-load guard is grad-only, even in eval-with-grad.
        assert result["expert_load_update"] == grad_enabled
        assert result["training_metric_update"] == (training and grad_enabled)
        assert result["inference_metric_update"] == (not training and not grad_enabled)
    assert {k: v for k, v in base.items() if k != "topk_request"} == {
        k: v for k, v in fixed.items() if k != "topk_request"}


@pytest.mark.parametrize("commit_name", ["BASE", "FIX"])
@pytest.mark.parametrize("training_metrics,inference_metrics", list(itertools.product(("", "mbs"), repeat=2)))
def test_native_metric_opt_outs_and_inference_guards(probe, commit_name, training_metrics, inference_metrics):
    tree = native_tree(probe, getattr(probe, commit_name))
    for training, grad in itertools.product((False, True), repeat=2):
        result = probe.conditions(tree, training=training, grad_enabled=grad, method="histogram",
                                  training_metrics=training_metrics, inference_metrics=inference_metrics)
        assert result["training_metric_update"] == bool(training and grad and training_metrics)
        assert result["inference_metric_update"] == bool(not training and not grad and inference_metrics)


def test_histogram_checkpoint_calls_request_same_k_only_after_fix(probe):
    for commit, requests in ((probe.BASE, [8, 9]), (probe.FIX, [9, 9])):
        tree = native_tree(probe, commit)
        actual = [probe.conditions(tree, training=True, grad_enabled=grad, method="histogram")
                  for grad in (False, True)]
        assert [row["topk_request"] for row in actual] == requests
        assert [row["statistics_update"] for row in actual] == [False, True]
    # Equal k requests are a condition regression, not evidence about CPU/HIP
    # tie ordering or equality of native routing maps/probabilities/gradients.


@pytest.mark.parametrize("commit_name", ["BASE", "FIX"])
def test_execute_actual_histogram_and_load_blocks_once_per_replayed_microbatch(probe, commit_name):
    tree = native_tree(probe, getattr(probe, commit_name))

    class Counter:
        def __init__(self):
            self.value = 0
            self.calls = 0

        def add_(self, value):
            self.value += value
            self.calls += 1
            return self

        def __iadd__(self, value):
            return self.add_(value)

    class Values:
        def __getitem__(self, key):
            assert key == (slice(None), -1)
            return boundary

    histogram, load = Counter(), Counter()
    scores, boundary, beta = object(), object(), object()
    config = SimpleNamespace(moe_router_quantile_balancing_method="histogram",
                             moe_router_quantile_balancing_num_bins=1000)
    router = SimpleNamespace(training=True, config=config, qb_histogram=histogram,
                             local_tokens_per_expert=load, qb_beta=beta)
    histogram_calls = []

    def histogram_sentinel(*args, padding_mask):
        # Sentinel rows measure invocation/accumulation ONLY, not native bins.
        assert args == (scores, boundary, beta, 1000) and padding_mask is None
        histogram_calls.append(1)
        return 4096

    load_method = probe.one(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "_apply_expert_bias")
    for microbatch in range(2):
        for grad in (False, True):
            namespace = {"self": router, "torch": SimpleNamespace(is_grad_enabled=lambda: grad,
                         no_grad=nullcontext), "padding_mask": None, "scores": scores,
                         "topk_result": SimpleNamespace(values=Values()),
                         "compute_qb_histogram": histogram_sentinel,
                         "routing_map": SimpleNamespace(sum=lambda dim: 8 * 4096)}
            statements = [probe.assignment(tree, name) for name in (
                "should_update_beta", "use_histogram", "use_marin", "use_exact")]
            statements += [probe.statistics(tree), *load_method.body[1:]]
            exec(compile(ast.Module(body=statements, type_ignores=[]), "<native-accounting>", "exec"), namespace)
            updates = microbatch + int(grad)
            assert histogram.calls == load.calls == len(histogram_calls) == updates
            assert histogram.value == updates * 4096
            assert load.value == updates * 8 * 4096


@pytest.mark.parametrize("commit_name", ["BASE", "FIX"])
def test_corrupt_or_unknown_excerpt_is_rejected_before_compilation(probe, tmp_path, commit_name):
    commit = getattr(probe, commit_name)
    altered = tmp_path / "router.py.txt"
    altered.write_text(BASE_NATIVE if commit == probe.BASE else FIX_NATIVE)
    assert ast.dump(probe.load_excerpt(altered, commit)) == ast.dump(native_tree(probe, commit))
    altered.write_bytes((BASE_NATIVE if commit == probe.BASE else FIX_NATIVE).encode() + b"\n")
    with pytest.raises(ValueError, match="untrusted"):
        probe.load_excerpt(altered, commit)
    with pytest.raises(ValueError, match="untrusted"):
        probe.load_excerpt(altered, "0" * 40)


@pytest.mark.parametrize("damage", ["parent", "files", "checksum"])
def test_git_source_verification_fails_closed_without_worktree_reads(probe, monkeypatch, damage):
    calls = []

    def git(argv):
        calls.append(argv[3:])
        assert argv[:3] == ["git", "-C", "/unread-worktree"]
        if argv[3] == "rev-parse":
            return (("0" * 40 if damage == "parent" else probe.BASE) + "\n").encode()
        if argv[3] == "diff-tree":
            return (probe.ROUTER + ("\nother.py" if damage == "files" else "") + "\n").encode()
        assert argv[3:] == ["show", f"{probe.BASE}:{probe.ROUTER}"]
        return b"untrusted router body"

    monkeypatch.setattr(probe.subprocess, "check_output", git)
    with pytest.raises(ValueError, match={"parent": "parent", "files": "outside", "checksum": "checksum"}[damage]):
        probe.verify_sources(Path("/unread-worktree"))
    assert len(calls) == {"parent": 1, "files": 2, "checksum": 3}[damage]


def test_missing_or_ambiguous_ast_targets_are_rejected(probe):
    with pytest.raises(ValueError, match="found 0"):
        probe.assignment(ast.parse("unrelated = True"), "should_update_beta")
    with pytest.raises(ValueError, match="found 2"):
        probe.assignment(ast.parse("should_update_beta = True\nshould_update_beta = False"), "should_update_beta")
