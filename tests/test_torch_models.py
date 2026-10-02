import numpy as np
import pytest

torch = pytest.importorskip("torch")

from traffic_ml.models.torch_models import make_torch_model


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer", "stgnn"])
def test_torch_model_output_shape(name):
    batch, steps, nodes, features = 3, 12, 4, 6
    x = torch.randn(batch, steps, nodes, features)
    adjacency = torch.eye(nodes)

    model = make_torch_model(
        name,
        num_nodes=nodes,
        num_features=features,
        adjacency=adjacency,
    )
    y = model(x)
    assert tuple(y.shape) == (batch, nodes)
