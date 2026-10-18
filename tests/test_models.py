import torch

from cryptogo.models import SEQUENCE_MODELS, ConditionalReturnFlow, CrossAssetFlow, CrossAssetTransformer


def test_sequence_models_emit_one_score_per_window():
    x = torch.randn(4, 48, 6)
    static = torch.randn(4, 4)
    for name, cls in SEQUENCE_MODELS.items():
        model = cls()
        model.eval()
        with torch.inference_mode():
            out = model(x, static)
        assert out.shape == (4,), name


def test_cross_asset_model_masks_padding():
    x = torch.randn(2, 5, 48, 6)
    static = torch.randn(2, 5, 4)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[0, -1] = True
    model = CrossAssetTransformer()
    model.eval()
    with torch.inference_mode():
        out = model(x, static, pad)
    assert out.shape == (2, 5)


def test_shallow_tail_keeps_extreme_noise_near_the_data_bound():
    flow = ConditionalReturnFlow(context_dim=4, bins=8, data_bound=0.30, noise_bound=3.0, tail_slope=0.05)
    context = torch.zeros(1, 4)
    z = torch.tensor([6.0])
    y = flow._transform(z, context, inverse=True)[0]
    # 0.30 + 0.05 * (6 - 3)
    assert torch.allclose(y, torch.tensor([0.45]), atol=1e-5)
    restored = flow._transform(y, context, inverse=False)[0]
    assert torch.allclose(restored, z, atol=1e-4)


def test_conditional_spline_inverts_and_matches_its_jacobian():
    torch.manual_seed(0)
    flow = ConditionalReturnFlow(context_dim=4, bins=8, bound=2.0, noise_bound=2.0, tail_slope=1.0).double()
    context = torch.randn(6, 4, dtype=torch.float64)
    y = torch.empty(6, dtype=torch.float64).uniform_(-1.2, 1.2)
    z, logdet = flow._transform(y, context, inverse=False)
    restored = flow._transform(z, context, inverse=True)[0]
    assert torch.allclose(restored, y, atol=1e-4, rtol=1e-4)
    eps = 1e-4
    slope = (
        flow._transform(y + eps, context, inverse=False)[0] - flow._transform(y - eps, context, inverse=False)[0]
    ) / (2 * eps)
    assert torch.allclose(slope.abs().log(), logdet, atol=1e-3, rtol=1e-3)
    assert abs(float(flow.quad_w.sum()) - 1.0) < 1e-5


def test_cross_flow_density_trains_and_probabilities_move():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 48, 6)
    static = torch.randn(2, 5, 4)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    y = torch.randn(2, 5) * 0.05
    model = CrossAssetFlow()
    model.train()
    log_p = model.log_prob(x, static, pad, y)
    assert log_p.shape == (2, 5)
    assert torch.isfinite(log_p).all()
    log_p.sum().backward()
    assert model.trunk.rnn.weight_ih_l0.grad is not None
    assert torch.isfinite(model.trunk.rnn.weight_ih_l0.grad).all()
    model.eval()
    with torch.inference_mode():
        mean = model.mean(x, static, pad)
        median = model.median(x, static, pad)
        q25 = model.quantile(x, static, pad, 0.25)
        q75 = model.quantile(x, static, pad, 0.75)
        low = model.prob_above(-1.0, x, static, pad)
        high = model.prob_above(1.0, x, static, pad)
    assert mean.shape == (2, 5)
    assert median.shape == (2, 5)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(median).all()
    assert torch.all(q25 < median)
    assert torch.all(median < q75)
    assert torch.all(low > high)
