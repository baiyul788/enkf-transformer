import torch
import numpy as np
from transformer_lorenz96 import TransformerModel


def apply_transformer_correction(xa: np.ndarray, model_path: str, k: int = 5) -> np.ndarray:
    xa_ds = xa[:, ::k]

    from config.experiment_config import get_default_inference_config
    config = get_default_inference_config()
    model_config = config.get_model_config()
    model = TransformerModel(**model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        inp = torch.FloatTensor(xa_ds.T).unsqueeze(0)
        pred_residual = model(inp).squeeze(0).T.numpy()

    xa_corrected = xa_ds + pred_residual
    return xa_corrected
