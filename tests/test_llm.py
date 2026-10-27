"""A language model across miners: exact int8 layers, checked work, speculative decoding,
and cheaters caught. A tiny random model in the Qwen2 layout stands in for a real one."""

import json
import struct

import numpy as np
import pytest

from green.llm.model import FP, Model
from green.llm.pipeline import CheatingStage, Pipeline, even_splits, local_stages, lookup_draft

CFG = {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
       "intermediate_size": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
       "vocab_size": 300}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    rng = np.random.default_rng(3)
    hs, inter, kv = CFG["hidden_size"], CFG["intermediate_size"], CFG["num_key_value_heads"] * 64
    t = {"model.embed_tokens.weight": rng.normal(0, 1, (CFG["vocab_size"], hs)),
         "model.norm.weight": np.ones(hs)}
    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name, shape in (("self_attn.q_proj.weight", (hs, hs)), ("self_attn.k_proj.weight", (kv, hs)),
                            ("self_attn.v_proj.weight", (kv, hs)), ("self_attn.o_proj.weight", (hs, hs)),
                            ("mlp.gate_proj.weight", (inter, hs)), ("mlp.up_proj.weight", (inter, hs)),
                            ("mlp.down_proj.weight", (hs, inter))):
            t[p + name] = rng.normal(0, 0.06, shape)
        for name, n in (("self_attn.q_proj.bias", hs), ("self_attn.k_proj.bias", kv), ("self_attn.v_proj.bias", kv)):
            t[p + name] = rng.normal(0, 0.1, n)
        t[p + "input_layernorm.weight"] = np.ones(hs)
        t[p + "post_attention_layernorm.weight"] = np.ones(hs)
    header, blobs, off = {}, [], 0
    for name, a in t.items():
        b = a.astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    (d / "config.json").write_text(json.dumps(CFG))
    return Model(str(d))


def run_layer(model, i, n_tokens=5, past_tokens=7, seed=0):
    rng = np.random.default_rng(seed)
    h0 = rng.integers(-2 * FP, 2 * FP, (past_tokens, CFG["hidden_size"]))
    first = model.forward(i, h0, np.arange(past_tokens), [])
    past = [(first["x1q"], first["s1"], first["yk"], first["yv"], np.arange(past_tokens))]
    h = rng.integers(-2 * FP, 2 * FP, (n_tokens, CFG["hidden_size"]))
    return model.forward(i, h, np.arange(past_tokens, past_tokens + n_tokens), past), past


def test_a_layers_honest_work_passes_and_any_change_is_caught(model):
    r = np.random.default_rng(9).integers(0, 1 << 18, 8192)
    sk = model.sketches(1, r)
    rec, past = run_layer(model, 1)
    assert model.check(1, rec, past, r, sk) is None
    assert model.forward(1, rec["h_in"], rec["pos"], past)["h_out"].tobytes() == rec["h_out"].tobytes()  # deterministic
    for key, change in (("yq", 1), ("yd", -1), ("yg", 3), ("h_out", 7), ("h_mid", 2), ("x1q", 5), ("mq", 9),
                        ("aq", 4), ("sa", None), ("s2", None)):
        bad = {k: v.copy() for k, v in rec.items()}
        if change is None:
            bad[key] = bad[key] * 1.01                                  # a scale, a little off
        else:
            flat = bad[key].reshape(-1)
            flat[len(flat) // 2] += change                              # one number, a little off
        assert model.check(1, bad, past, r, sk) is not None, key
    bad_past = [tuple(a.copy() for a in past[0])]
    bad_past[0][2].reshape(-1)[0] += 1                                  # a cache row's product
    assert model.check(1, rec, bad_past, r, sk) is not None
    # rounding by one unit is allowed: another machine's floating point may land there
    off = {k: v.copy() for k, v in rec.items()}
    off["h_out"].reshape(-1)[0] += 1
    assert model.check(1, off, past, r, sk) is None


def test_the_pipeline_gives_the_models_own_tokens_and_speculation_changes_nothing(model):
    stages = local_stages(model, even_splits(4, 2))
    pipe = Pipeline(model, [s.info for s in stages], opens=2)
    try:
        prompt = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6]
        plain, st1 = pipe.generate(prompt, max_new=12, draft=0)
        spec, st2 = pipe.generate(prompt, max_new=12, draft=4)
        pipe.finish_audits()
        assert plain == spec and st1["passes"] == 12 and st2["passes"] <= 12
        # the same tokens as running every layer here, in one go
        h, pos = model.embed_tokens(prompt + plain[:-1]), np.arange(len(prompt) + 11)
        for i in range(4):
            h = model.forward(i, h, pos, [])["h_out"]
        assert [int(t) for t in np.argmax(model.logits(h)[len(prompt) - 1:], axis=1)] == plain
        assert len([a for a in pipe.audit_log if a[0] == "ok"]) == 2 * 2 * (st1["passes"] + st2["passes"])
    finally:
        pipe.stop()
        for s in stages:
            s.stop()


def test_a_stage_that_skips_a_layer_is_caught_and_named(model):
    stages = local_stages(model, even_splits(4, 2), cheat={1: "skip"})
    pipe = Pipeline(model, [s.info for s in stages], opens=2)      # every layer of a 2-layer stage, every pass
    try:
        with pytest.raises(CheatingStage) as e:
            pipe.generate([1, 2, 3, 4], max_new=6, draft=0)
            pipe.finish_audits()
        assert e.value.stage["layers"] == [2, 4] and "layer 3" in str(e.value)
    finally:
        pipe.stop()
        for s in stages:
            s.stop()


def test_lookup_draft_copies_what_followed_the_same_tokens():
    assert lookup_draft([1, 2, 3, 9, 9, 1, 2, 3], 3) == [9, 9, 1]
    assert lookup_draft([1, 2, 3], 3) == [] and lookup_draft([4, 4, 4], 0) == []
