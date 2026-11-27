"""Offline review checks; do not import or execute the unsigned Megatron candidate."""

import ast
import os
from pathlib import Path

import pytest


FLAG = "moe_recompute_post_combine"
MODELS = "megatron/core/models"
TRANSFORMER = "megatron/core/transformer"
CALLABLES = f"{MODELS}/gpt/fine_grained_callables.py"
MOE = f"{TRANSFORMER}/moe/moe_layer.py"
CONFIG = f"{TRANSFORMER}/transformer_config.py"
SCHEDULE = f"{MODELS}/common/model_chunk_schedule_plan.py"
UTILS = "megatron/core/pipeline_parallel/utils.py"


@pytest.fixture(scope="module")
def source():
    if not (path := os.environ.get("APERTUS_POST_COMBINE_SOURCE")):
        pytest.skip("set APERTUS_POST_COMBINE_SOURCE to the checkout under review")

    def parse(relative):
        return ast.parse((Path(path) / relative).read_text())

    return parse


def named(body, *names):
    for name in names:
        body = next(node for node in body.body if getattr(node, "name", None) == name)
    return body


def calls(node):
    return [
        (call.lineno, ast.unparse(call.func))
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]


def line(node, function):
    (number,) = (n for n, f in calls(node) if f == function)
    return number


def attributes(node):
    return {value.attr for value in ast.walk(node) if isinstance(value, ast.Attribute)}


def test_opt_in_inherits_native_config_restrictions(source):
    cfg = named(source(CONFIG), "TransformerConfig")
    setting = next(
        x for x in cfg.body if isinstance(x, ast.AnnAssign) and ast.unparse(x.target) == FLAG
    )
    assert ast.unparse(setting.annotation) == "bool"
    assert setting.value.value is False
    init = named(cfg, "__post_init__")
    gate = next(x for x in init.body if isinstance(x, ast.If) and FLAG in attributes(x.test))
    assert isinstance(gate.test, ast.BoolOp) and isinstance(gate.test.op, ast.And)
    assert ast.unparse(gate.test.values[0]) == f"self.{FLAG}"
    assert isinstance(gate.body[0], ast.Raise)
    assert {
        "moe_resident_fp8_fnuz", "moe_fp8_dispatch_fnuz", "tensor_model_parallel_size",
        "enable_autocast", "quant_recipe", "sandwich_norm", "normalization", "fused_residual_rmsnorm",
        "moe_shared_expert_intermediate_size", "moe_shared_expert_overlap", "mtp_num_layers",
        "recompute_granularity",
    } <= attributes(gate.test)
    native = next(
        x for x in init.body if isinstance(x, ast.If)
        and ast.unparse(x.test) == "self.moe_resident_fp8_fnuz"
    )
    assert {
        "delay_wgrad_compute", "overlap_moe_expert_parallel_comm", "gradient_accumulation_fusion",
        "cuda_graph_impl", "fp8", "fp4", "cpu_offloading", "fine_grained_activation_offloading",
    } <= attributes(native)
    factory = named(source("megatron/training/arguments.py"), "_add_network_size_args")
    assert FLAG not in {
        x.value for x in ast.walk(factory) if isinstance(x, ast.Constant) and isinstance(x.value, str)
    }  # Not excluded from the dataclass-generated CLI.


def test_boundary_is_pure_local_bf16_projection_and_add(source):
    layer = named(source(MOE), "MoELayer")
    post = named(layer, "postprocess")
    local = named(layer, "_project_and_add_shared")
    assert line(post, "self.token_dispatcher.combine_postprocess") < line(post, "checkpoint.checkpoint")
    assert [f for _, f in calls(local)] == ["self.fc2_latent_proj"]
    (add,) = (x for x in ast.walk(local) if isinstance(x, ast.BinOp))
    assert ast.unparse(add) == "output + shared_expert_output"
    assert {f for _, f in calls(post)} == {
        "self.token_dispatcher.combine_postprocess", "checkpoint.checkpoint",
        "self._project_and_add_shared",
    }


def test_checkpoint_is_per_forward_and_registered_before_residual(source):
    build = named(source(CALLABLES), "build_transformer_layer_callables")
    post = named(build, "submodule_post_combine_forward")
    assert (
        line(post, "tensor_parallel.CheckpointWithoutOutput")
        < line(post, "layer._apply_post_norm")
        < line(post, "checkpoint.discard_output_and_register_recompute")
        < line(post, "layer.bias_dropout_add_exec_handler")
    )
    gate = next(x for x in post.body if isinstance(x, ast.If) and FLAG in attributes(x.test))
    assert ast.unparse(gate.test) == f"layer.config.{FLAG} and layer.training and torch.is_grad_enabled()"
    assert ast.unparse(gate.orelse[0].value) == "layer.mlp.postprocess(output, shared_expert_output)"
    assert "node.layer_state.residual.record_stream" in {f for _, f in calls(post)}
    assert "shared_expert_output.record_stream" in {f for _, f in calls(post)}
    assert not any(
        isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Store) and "checkpoint" in x.attr
        for x in ast.walk(post)
    )
    assert not {"wait", "wait_event", "synchronize", "empty_cache", "backward_dw"} & attributes(post)


def test_real_plan_keeps_comm_stream_and_single_projection_drain(source):
    plan = named(source(SCHEDULE), "TransformerLayerSchedulePlan")
    run = named(plan, "run")
    build = named(plan, "_build_callable_nodes")
    assert (
        line(run, "b_layer.moe_post_combine.backward")
        < line(run, "b_layer.moe_combine.backward")
        < line(run, "b_layer.mlp.backward")
        < line(run, "b_layer.mlp.backward_dw")
    )
    for node in ("moe_combine", "moe_post_combine"):
        assignment = next(
            x for x in ast.walk(build) if isinstance(x, ast.Assign)
            and any(ast.unparse(t) == f"self.{node}" for t in x.targets)
        )
        assert ast.unparse(assignment.value.args[0]) == "comm_stream"
    region = named(source(UTILS), "ScheduleNode", "_backward")
    assert line(region, "self.stream_acquire_context") < line(region, "self.backward_func")
    dw = named(source(MOE), "MoELayer", "backward_dw")
    assert line(dw, "torch.cuda.stream") < line(dw, "self.fc2_latent_proj.backward_dw")
