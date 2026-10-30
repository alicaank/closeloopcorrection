import json
from pathlib import Path

_EXTRA = {}


def profiles():
    return dict(
        json.loads(Path(__file__).with_name("profiles.json").read_text()), **_EXTRA
    )


def profile(model):
    found = profiles()
    if model not in found:
        raise ValueError(f'Unknown model {model!r}; available: {", ".join(found)}')
    return found[model]


def register_model(
    name, *, direction="c2w", axes="opencv", scale_strategy="native_callback"
):
    if name in profiles():
        raise ValueError("Model already registered")
    if (
        direction not in ["c2w", "w2c"]
        or axes not in ["opencv", "opengl"]
        or scale_strategy
        not in ["anchor_depth", "metric", "metric_div4", "native_callback"]
    ):
        raise ValueError("Invalid native camera contract")
    _EXTRA[name] = dict(
        label=name,
        direction=direction,
        axes=axes,
        scale_strategy=scale_strategy,
        status="user_hook",
        notes="User-supplied native integration; GPU validation required",
    )
