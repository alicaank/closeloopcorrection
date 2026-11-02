"""CPU-only import and preflight for the legacy adapters' explicit batch contract."""

import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import PixelTransform, poses
from .registry import profile

# Required recorded poses for the default native schedules in integrations.command.
REQUIREMENTS = {
    "lyra": (481, 481),
    "spatia": (481, 481),
    "matrixgame": (1, 505),
    "vmem": (41, 41),
    "astronex": (77, 93),
    "lingbot": (1, 173),
    "dreamx": (1, 165),
    "worldwarp": (1, 161),
    "sana": (169, 169),
    "memorizon": (161, 161),
    "worldplay": (161, 173),
}
METRIC = {"lyra", "sana", "matrixgame"}


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", name
    ):
        raise ValueError(
            "Scene IDs must contain only letters, digits, underscores and hyphens"
        )
    return name


def contained(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"Batch path escapes its root: {relative}")
    return p


def positive(value, label):
    if not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def geometry(arrays, size):
    w, h = size
    main = poses(arrays["w2c"])
    extended = poses(arrays.get("w2c_overrun", main))
    if (
        main.ndim != 3
        or len(main) < 1
        or extended.ndim != 3
        or len(extended) < len(main)
    ):
        raise ValueError("Expected nonempty w2c and an overrun at least as long")
    if not np.allclose(main[0], np.eye(4), atol=1e-5):
        raise ValueError("Poses must already be anchor-relative OpenCV w2c")
    if not np.allclose(extended[: len(main)], main, atol=1e-6):
        raise ValueError("w2c_overrun must preserve the recorded w2c prefix")
    k = np.asarray(arrays["intrinsics"], dtype=float)
    if k.shape == (3, 3):
        k = np.repeat(k[None], len(main), axis=0)
    if k.shape != (len(main), 3, 3) or not np.isfinite(k).all():
        raise ValueError("intrinsics must be 3x3 or one pixel K per w2c frame")
    if not np.allclose(k, k[0], atol=1e-5, rtol=0):
        raise ValueError(
            "Legacy adapters use a fixed K; variable intrinsics require a new adapter"
        )
    if (
        not np.allclose(k[:, 2], [0, 0, 1])
        or np.any(k[:, [0, 1], [0, 1]] <= 0)
        or not np.allclose(k[:, [0, 1], [1, 0]], 0)
    ):
        raise ValueError(
            "Expected positive focal lengths, zero skew and pinhole pixel intrinsics"
        )
    depth = np.asarray(arrays["anchor_depth"], dtype=float)
    if depth.shape != (h, w):
        raise ValueError("Registered anchor depth and source image dimensions differ")
    good = np.isfinite(depth) & (depth > 0)
    if good.sum() < 256 or good.mean() < 0.2:
        raise ValueError("Insufficient valid anchor depth")
    for key, expected in [("image_width", w), ("image_height", h)]:
        if key in arrays and int(arrays[key]) != expected:
            raise ValueError(f"{key} disagrees with image")
    return dict(
        w2c=main,
        w2c_overrun=extended,
        intrinsics=k,
        anchor_depth=depth,
        image_width=np.int64(w),
        image_height=np.int64(h),
    )


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def captions(value):
    if not isinstance(value, dict) or not value or "0" not in value:
        raise ValueError('Captions must include frame key "0"')
    if any(
        not re.fullmatch(r"0|[1-9][0-9]*", str(k))
        or not isinstance(v, str)
        or not v.strip()
        for k, v in value.items()
    ):
        raise ValueError(
            "Caption keys must be nonnegative frame IDs and values nonempty text"
        )
    return value


def index_rows(batch, scenes=None):
    batch = Path(batch)
    index = json.loads((batch / "index.json").read_text())
    rows = index["scenes"]
    names = [safe_name(r["name"]) for r in rows]
    if not names or len(set(names)) != len(names):
        raise ValueError("Scene IDs must be nonempty and unique")
    if scenes:
        if len(set(scenes)) != len(scenes):
            raise ValueError("Duplicate selected scenes")
        if set(scenes) - set(names):
            raise ValueError(
                "Unknown scenes: " + ", ".join(sorted(set(scenes) - set(names)))
            )
        rows = [next(r for r in rows if r["name"] == n) for n in scenes]
    return index, rows


def validate_batch(batch, model=None, scenes=None):
    """Validate CPU inputs, not model installation or inferred geometry accuracy."""
    batch = Path(batch).resolve()
    if model:
        profile(model)
    index, rows = index_rows(batch, scenes)
    required = REQUIREMENTS.get(model, (1, 1))
    result = dict(
        passed=True,
        model=model,
        scenes=[],
        dataset=index.get("dataset"),
        split=index.get("split"),
        warnings=[],
    )
    if not index.get("dataset") or not index.get("split"):
        result["warnings"].append(
            "Dataset/split provenance missing; this is not a held-out split verification"
        )
    for row in rows:
        name = row["name"]
        errors = []
        try:
            image = contained(batch, row["image"])
            if image != contained(batch, f"images/{name}.png"):
                raise ValueError("Legacy launchers require images/SCENE.png")
            with Image.open(image) as im:
                size = im.size
            if size != (864, 480):
                raise ValueError(
                    "Legacy adapters require 864x480; use import-batch --resize with registered geometry"
                )
            a = read_arrays(contained(batch, f"trajectories/{name}.npz"))
            for key in ["w2c_overrun", "image_width", "image_height"]:
                if key not in a:
                    raise ValueError("Missing array: " + key)
            g = geometry(a, size)
            if len(g["w2c"]) < required[0] or len(g["w2c_overrun"]) < required[1]:
                raise ValueError(
                    f'{model} needs at least {required[0]} main / {required[1]} overrun poses; got {len(g["w2c"])} / {len(g["w2c_overrun"])}'
                )
            caps = captions(
                json.loads(contained(batch, f"captions/{name}.json").read_text())
            )
            if not isinstance(row.get("caption"), str) or not row["caption"].strip():
                raise ValueError("Missing scene caption")
            if model in METRIC:
                positive(row["metres_per_anchor_unit"], "metres_per_anchor_unit")
            if "metres_per_anchor_unit" in a and "metres_per_anchor_unit" in row:
                if not np.isclose(
                    float(a["metres_per_anchor_unit"]),
                    float(row["metres_per_anchor_unit"]),
                    rtol=1e-6,
                ):
                    raise ValueError("Index and NPZ metric conversions disagree")
            if model in {"memorizon", "worldplay", "astronex"}:
                prompt = row.get("prompts", {}).get(model)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Missing prompts.{model}")
                if model == "memorizon" and not prompt.startswith(
                    "First-person perspective — character not visible. "
                ):
                    raise ValueError(
                        "Memorizon prompt needs the native first-person prefix"
                    )
            if model == "worldplay" and not np.allclose(
                g["intrinsics"][0, :2, 2], [432, 240], atol=1e-5
            ):
                raise ValueError(
                    "Historical WorldPlay normalization assumes a centered principal point"
                )
            # Astronex/WorldPlay units are identified at runtime, never from
            # recorded native poses or scalar metadata supplied by the batch.
            result["scenes"].append(
                dict(
                    name=name,
                    passed=True,
                    frames=len(g["w2c"]),
                    overrun_frames=len(g["w2c_overrun"]),
                    captions=len(caps),
                )
            )
        except (ValueError, KeyError, OSError, TypeError) as e:
            errors.append(str(e))
            result["passed"] = False
            result["scenes"].append(dict(name=name, passed=False, errors=errors))
    return result


def fingerprint(batch, scenes=None):
    batch = Path(batch)
    _, rows = index_rows(batch, scenes)
    h = hashlib.sha256((batch / "index.json").read_bytes())
    for row in rows:
        for rel in [
            row["image"],
            f'trajectories/{row["name"]}.npz',
            f'captions/{row["name"]}.json',
        ]:
            h.update(rel.encode())
            h.update(contained(batch, rel).read_bytes())
    return h.hexdigest()


def import_batch(manifest, out, resize=False):
    """Import supplied geometry, never estimate/extrapolate cameras or invent a scale."""
    source = Path(manifest).resolve()
    spec = json.loads(source.read_text())
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Batch output exists; use a new directory")
    if (
        spec.get("pose_convention") != "opencv_w2c_anchor_relative"
        or spec.get("depth_convention") != "z_depth_same_units_as_translation"
    ):
        raise ValueError(
            "Declare pose_convention and depth_convention explicitly; see dataset guide"
        )
    if not spec.get("dataset") or not spec.get("split"):
        raise ValueError("Record dataset and split provenance")
    names = [safe_name(r["name"]) for r in spec["scenes"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Scene IDs must be nonempty and unique")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".camera-import-", dir=out.parent) as td:
        stage = Path(td) / "batch"
        stage.mkdir()
        for folder in ["images", "trajectories", "captions"]:
            (stage / folder).mkdir()
        rows = []
        for row in spec["scenes"]:
            name = row["name"]
            a = read_arrays(source.parent / row["geometry"])
            with Image.open(source.parent / row["image"]) as im:
                image = im.convert("RGB")
            g = geometry(a, image.size)
            original_size = image.size
            if original_size != (864, 480):
                if not resize:
                    raise ValueError(
                        "Input grid is not 864x480; explicit --resize required"
                    )
                transform = PixelTransform(original_size, (864, 480))
                g["intrinsics"] = np.stack(
                    [transform.intrinsics(k) for k in g["intrinsics"]]
                )
                g["anchor_depth"] = transform.depth(g["anchor_depth"])
                image = image.resize((864, 480), Image.Resampling.BILINEAR)
                g["image_width"] = np.int64(864)
                g["image_height"] = np.int64(480)
            cap = captions(row.get("captions", {"0": row["caption"]}))
            record = {
                k: v
                for k, v in row.items()
                if k not in ["geometry", "image", "captions"]
            }
            record["image"] = f"images/{name}.png"
            if "metres_per_anchor_unit" in row:
                g["metres_per_anchor_unit"] = positive(
                    row["metres_per_anchor_unit"], "metres_per_anchor_unit"
                )
            if "native_w2c_latent" in a:
                g["native_w2c_latent"] = a["native_w2c_latent"]
            image.save(stage / record["image"])
            np.savez_compressed(stage / "trajectories" / f"{name}.npz", **g)
            (stage / "captions" / f"{name}.json").write_text(json.dumps(cap, indent=2))
            rows.append(record)
        index = {k: v for k, v in spec.items() if k != "scenes"}
        index["scenes"] = rows
        (stage / "index.json").write_text(json.dumps(index, indent=2))
        report = validate_batch(stage)
        if not report["passed"]:
            raise ValueError(json.dumps(report))
        (stage / "import_receipt.json").write_text(
            json.dumps(
                dict(
                    source_manifest_sha256=hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    resize=resize,
                    batch_sha256=fingerprint(stage),
                ),
                indent=2,
            )
        )
        stage.rename(out)
    return dict(batch=str(out), scenes=names, batch_sha256=fingerprint(out))
