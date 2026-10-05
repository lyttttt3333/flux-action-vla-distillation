import torch

from flux_action.inference import sampling


def test_fused_four_step_unipc_matches_generic_solver(monkeypatch):
    for name in ("_unipc_step0", "_unipc_step1", "_unipc_step2", "_unipc_final"):
        compiled = getattr(sampling, name)
        monkeypatch.setattr(sampling, name, compiled._torchdynamo_orig_callable)

    generator = torch.Generator().manual_seed(123)
    samples = {
        "x_video": torch.randn(1, 31, 7, generator=generator),
        "x_action": torch.randn(1, 13, 3, generator=generator),
    }

    def predict(values, tick):
        time = tick.float() / 1000.0
        return {key: value * 0.125 + time for key, value in values.items()}

    expected = sampling.cosmos_unipc_order2(
        {key: value.clone() for key, value in samples.items()},
        predict,
        n_steps=4,
        shift=5.0,
    )
    actual = sampling.cosmos_unipc_order2_fused4(
        {key: value.clone() for key, value in samples.items()}, predict, shift=5.0
    )
    for key in samples:
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=2e-6)
