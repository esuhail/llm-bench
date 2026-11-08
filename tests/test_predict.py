import pytest

import predict


def gpu(name):
    g = predict.find_gpu(name)
    assert g, name
    return g


def test_find_gpu_prefers_longest_match():
    assert predict.find_gpu("NVIDIA GeForce RTX 4060 Ti")["name"] == "RTX 4060 Ti"
    assert predict.find_gpu("NVIDIA GeForce RTX 4060")["name"] == "RTX 4060"
    assert predict.find_gpu("Some Unknown Card") is None


@pytest.mark.parametrize("text,expected", [("9.0B", 9.0), ("270M", 0.27), ("3B", 3.0), ("", None), (None, None)])
def test_params_from_text(text, expected):
    got = predict.params_from_text(text)
    assert got == pytest.approx(expected) if expected else got is None


def test_speed_increases_with_bandwidth_and_drops_with_size():
    assert predict.speed(5, 0.55, 500) > predict.speed(5, 0.55, 250)
    assert predict.speed(2, 0.55, 500) > predict.speed(8, 0.55, 500)


def test_speed_is_capped_by_per_token_overhead():
    assert predict.speed(0.001, 1, 10_000) < 1 / predict.PER_TOKEN_OVERHEAD_S


def test_kv_cache_grows_with_context_and_size():
    assert predict.kv_cache_gb(8, 16384) == pytest.approx(2 * predict.kv_cache_gb(8, 8192))
    assert predict.kv_cache_gb(70, 8192) > predict.kv_cache_gb(8, 8192)


def test_memory_breakdown_adds_up():
    m = predict.find_model("llama3.1:8b")
    mem = predict.memory_breakdown(m)
    assert mem["total_gb"] == pytest.approx(mem["weights_gb"] + mem["kv_gb"] + mem["runtime_gb"])


def test_fewer_bits_means_less_memory():
    m = predict.find_model("llama3.1:8b")
    assert predict.memory_breakdown(m, bits=3)["total_gb"] < predict.memory_breakdown(m, bits=8)["total_gb"]


def test_small_model_fits_and_huge_model_does_not():
    g = gpu("RTX 4060")
    assert predict.predict(g, predict.find_model("llama3.2:3b"), 0.55)["fits"]
    assert not predict.predict(g, predict.find_model("llama3.3:70b"), 0.55)["fits"]


def test_moe_runs_faster_than_dense_of_same_size():
    g = gpu("RTX 4090")
    moe = next(m for m in predict.MODELS if m["active_params_b"] < m["params_b"] / 2 and m["params_b"] < 40)
    dense = dict(moe, active_params_b=moe["params_b"])
    assert predict.predict(g, moe, 0.55)["tokens_per_sec"] > predict.predict(g, dense, 0.55)["tokens_per_sec"]


def test_efficiency_round_trips():
    """A result measured at exactly the predicted speed should give back the same efficiency."""
    g = gpu("RTX 4060")
    gb = predict.weights_gb(8.0)
    tps = predict.speed(gb, 0.6, g["bandwidth_gbs"])
    r = {"gpu": "NVIDIA GeForce RTX 4060", "model": "llama3.1:8b", "parameter_size": "8.0B", "tokens_per_sec": tps}
    assert predict.efficiency_of(r) == pytest.approx(0.6)


def test_efficiency_skips_spilled_runs():
    r = {"gpu": "RTX 4060", "model": "llama3.1:8b", "parameter_size": "8.0B", "tokens_per_sec": 20, "gpu_offload_pct": 60}
    assert predict.efficiency_of(r) is None


def test_learn_efficiency_defaults_without_data():
    eff, how = predict.learn_efficiency([], "RTX 4060")
    assert eff == predict.DEFAULT_EFFICIENCY and "default" in how


def test_predict_all_covers_the_catalog():
    out = predict.predict_all("RTX 4060", [])
    assert len(out["models"]) == len(predict.MODELS)
    assert predict.predict_all("Not a GPU", []) is None


def test_catalog_is_well_formed():
    names = [m["name"] for m in predict.MODELS]
    assert len(names) == len(set(names)), "duplicate model names"
    for m in predict.MODELS:
        assert 0 < m["active_params_b"] <= m["params_b"], m["name"]
        assert set(m["tags"]) <= set(predict.TAGS), m["name"]
    for g in predict.GPUS:
        assert g["vram_gb"] > 0 and g["bandwidth_gbs"] > 0, g["name"]
