import torch
from torch import nn
from transformers import AutoModel


class Ada_REVE(nn.Module):
    def __init__(self, args, ch_names=None):
        super().__init__()
        self.main_model = AutoModel.from_pretrained(
            args.reve_model_id,
            trust_remote_code=True,
        )
        self.task_head = nn.Identity()
        self.reve_pool = args.reve_pool
        self._init_positions(args.reve_pos_id, ch_names)

    def _init_positions(self, pos_id, ch_names):
        if not ch_names:
            self.positions = None
            return
        try:
            pos_model = AutoModel.from_pretrained(pos_id, trust_remote_code=True)
        except Exception:
            self.positions = None
            return

        positions = self._build_positions(pos_model, ch_names)
        if positions is None:
            self.positions = None
            return

        positions = torch.as_tensor(positions, dtype=torch.float32)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        elif positions.dim() != 3:
            positions = positions.reshape(1, positions.shape[0], -1)
        self.register_buffer("positions", positions, persistent=False)

    def _build_positions(self, pos_model, ch_names):
        for attr_name in ("get_positions", "positions_from_ch_names", "get_position"):
            if hasattr(pos_model, attr_name):
                try:
                    return getattr(pos_model, attr_name)(ch_names)
                except Exception:
                    pass
        try:
            return pos_model(ch_names)
        except Exception:
            pass

        if hasattr(pos_model, "positions"):
            positions = getattr(pos_model, "positions")
            if isinstance(positions, dict):
                coords = []
                for name in ch_names:
                    if name not in positions:
                        return None
                    coords.append(torch.as_tensor(positions[name]))
                return torch.stack(coords, dim=0)
        return None

    def _call_backbone(self, x, positions):
        try:
            return self.main_model(x, positions=positions)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword" in message or "pos" in message or "positions" in message:
                try:
                    return self.main_model(x, positions)
                except TypeError:
                    if positions is None:
                        return self.main_model(x, None)
            if positions is None and "missing 1 required positional argument" in message:
                return self.main_model(x, None)
            raise

    def _extract_features(self, output):
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        if torch.is_tensor(output):
            return output
        raise ValueError("Unsupported output type from REVE backbone.")

    def _pool_features(self, features):
        if features.dim() == 3:
            if self.reve_pool == "mean":
                features = features.mean(dim=1)
            elif self.reve_pool == "first":
                features = features[:, 0]
            else:
                raise ValueError(f"Unsupported reve_pool: {self.reve_pool}")
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() > 2:
            features = features.flatten(1)
        return features

    def forward(self, x):
        positions = None
        if hasattr(self, "positions") and self.positions is not None:
            positions = self.positions.to(device=x.device, dtype=x.dtype)
            if positions.dim() == 3 and positions.shape[0] == 1:
                positions = positions.expand(x.shape[0], -1, -1)
        output = self._call_backbone(x, positions)
        features = self._extract_features(output)
        features = self._pool_features(features)
        return self.task_head(features)
