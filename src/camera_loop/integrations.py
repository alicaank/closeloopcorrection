"""Materialize isolated legacy/native adapters from source-only, hash-checked templates."""

import hashlib, json, os, re, subprocess
from pathlib import Path
from .registry import profiles, profile

BUNDLE = Path(__file__).with_name("bundled")
A = Path("analysis/plan_base_restart_20261008")
N = A / "new_models_loop_20261028"
C = A / "continuous_eval_20261029"
L = A / "long_domain_20261028"
F = A / "fair_common_20261025"
V = C / "scale_20261030/correction_v3_20261030"


def materialize(workspace, config, clone=False):
    w = Path(workspace).resolve()
    if w.exists() and any(w.iterdir()):
        raise ValueError(
            "Use an empty workspace; never materialize over experiments or installed models"
        )
    selected = config.get("models", list(profiles()))
    for m in selected:
        if "repo" not in profile(m):
            raise ValueError(f"{m} is a custom hook; supply its host model directly")
    values = {
        "@WORKSPACE@": str(w),
        "@HOME_ROOT@": str(Path(config.get("home_root", Path.home())).resolve()),
        "@CACHE_ROOT@": str(
            Path(config.get("cache_root", Path.home() / ".cache")).resolve()
        ),
        "@CONDA_ROOT@": str(
            Path(config.get("conda_root", Path.home() / "miniconda3")).resolve()
        ),
    }
    overrides = config.get("path_map", {})
    mappings = dict(values, **overrides)
    # The same model interpreter may also launch native subprocesses (e.g. Spatia
    # MapAnything/rendering). A Python override must relocate those references too.
    for model, interpreter in config.get("python", {}).items():
        native = profile(model)["python"]
        key = native if native.startswith("@") else "@WORKSPACE@/" + native
        if not Path(interpreter).is_absolute():
            raise ValueError("Python overrides must be absolute paths")
        if key in overrides and overrides[key] != interpreter:
            raise ValueError(f"Conflicting interpreter/path_map override for {model}")
        mappings[key] = interpreter
    for key, value in mappings.items():
        if not isinstance(value, str) or any(
            c in value for c in ["\n", "\r", "'", '"', "\\"]
        ):
            raise ValueError(
                "Path mappings must be plain paths without quotes/backslashes/newlines"
            )
    pattern = "|".join(re.escape(k) for k in sorted(mappings, key=len, reverse=True))

    def render(s):
        return re.sub(pattern, lambda m: mappings[m.group()], s)

    entries = json.loads((BUNDLE / "manifest.json").read_text())
    # Verify all inputs before creating files or cloning anything.
    for item in entries:
        data = (BUNDLE / "templates" / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Bundled source hash mismatch: " + item["path"])
    for path, expected in json.loads(
        (BUNDLE / "patch_manifest.json").read_text()
    ).items():
        if hashlib.sha256((BUNDLE / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Bundled patch hash mismatch: " + path)
    w.mkdir(parents=True, exist_ok=True)
    pins = json.loads((BUNDLE / "repos.json").read_text())
    clone_models = set(selected) | ({"lyra", "vipe"} if selected else set())
    if clone:
        for model in sorted(clone_models):
            p = pins[model]
            target = w / p["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    p["url"],
                    str(target),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", p["commit"]],
                check=True,
            )
            if model == "lyra":
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(target),
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                        "--",
                        "Lyra-2/lyra_2/_src/inference/depth_anything_3",
                        "Lyra-2/lyra_2/_src/inference/vipe",
                    ],
                    check=True,
                )
            elif (target / ".gitmodules").exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(target),
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                    ],
                    check=True,
                )
            patch = render((BUNDLE / p["patch"]).read_text())
            if patch.strip():
                subprocess.run(
                    ["git", "-C", str(target), "apply", "--whitespace=nowarn", "-"],
                    input=patch,
                    text=True,
                    check=True,
                )
    for item in entries:
        src = BUNDLE / "templates" / item["path"]
        target = w / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        text = render(src.read_text())
        # Respect an explicit external cache root even where old adapters use Path.home().
        text = re.sub(
            r"Path\.home\(\)\s*/\s*['\"]\.cache['\"]",
            lambda m: "Path(" + repr(values["@CACHE_ROOT@"]) + ")",
            text,
        )
        if target.name == "fair_common.py":
            text = text.replace(
                "BATCH = HERE / 'batch_ours10'",
                "BATCH = Path(os.environ.get('CAMERA_BATCH', str(HERE / 'batch_ours10')))",
            )
        target.write_text(text)
    # Refresh versioned snapshot hashes after relocation; source identity is retained in the bundle manifest.
    manifest_path = w / V / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest.items():
        entry["sha256"] = hashlib.sha256(
            (w / V / "snapshots" / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    receipt = dict(
        models=selected,
        workspace=str(w),
        paths=values,
        python=config.get("python", {}),
        bundle_manifest_sha256=hashlib.sha256(
            (BUNDLE / "manifest.json").read_bytes()
        ).hexdigest(),
        cloned=clone,
    )
    (w / "camera_loop_workspace.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def command(
    workspace,
    model,
    python,
    batch,
    scene,
    out,
    arm="correction",
    device="0",
    quality="enforce",
):
    if "repo" not in profile(model):
        raise ValueError(f"{model} is a custom hook; invoke it through AlignedLoop")
    w = Path(workspace).resolve()
    b = str(Path(batch).resolve())
    o = str(Path(out).resolve())
    if arm not in ["native", "baseline", "sham", "correction", "replan_measured"]:
        raise ValueError("Unknown arm")
    if arm == "replan_measured" and model != "vmem":
        raise ValueError(
            "Measured-memory writes are a separate VMem ablation, not generic correction"
        )
    if model in ["memorizon", "worldplay"] and arm not in ["baseline", "correction"]:
        raise ValueError("Historical adapter has no verified native/sham controls")
    if model in ["lyra", "spatia", "vmem"] and arm == "native":
        raise ValueError("Use the released native entrypoint for native parity")
    if quality not in ["enforce", "report"]:
        raise ValueError("Invalid quality mode")
    common = ["--batch", b, "--scene", scene, "--out", o]
    corrected = {"correction": "replan_commanded"}.get(arm, arm)
    if model == "matrixgame":
        script = C / "mg35_stream_run.py"
        args = common + ["--arm", arm, "--blocks", "6", "--seed", "7"]
    elif model == "lyra":
        script = V / "lyra_v3.py"
        args = common + ["--arm", arm, "--frames", "481", "--observer-quality", quality]
    elif model == "spatia":
        script = V / "spatia_v3.py"
        args = common + [
            "--arms",
            corrected,
            "--seed",
            "7",
            "--num_frames",
            "481",
            "--first_round_frames",
            "121",
            "--round_frames",
            "81",
            "--hist_frames",
            "9",
            "--meas_stride",
            "12",
            "--observer-quality",
            quality,
            "--visibility-views-per-batch",
            "1",
        ]
    elif model == "worldwarp":
        script = V / "worldwarp_v3.py"
        args = common + [
            "--arm",
            arm,
            "--seed",
            "7",
            "--chunks",
            "2",
            "--scale_mode",
            "anchor",
            "--camera_window",
            "fixed",
            "--continuation-check",
            "enforce",
            "--observer-quality",
            quality,
        ]
    elif model == "sana":
        script = V / "sana_v3.py"
        args = common + [
            "--arm",
            arm,
            "--seed",
            "7",
            "--frames",
            "169",
            "--scale_mode",
            "anchor",
            "--observer-quality",
            quality,
        ]
    elif model == "vmem":
        script = L / "vmem_run.py"
        args = [
            "--out",
            o,
            "--arms",
            str(w / L / "vmem_arms.json"),
            "--arm",
            corrected,
            "--scenes",
            scene,
            "--frames",
            "40",
            "--first_step",
            "4",
            "--nms",
            "cfg",
            "--autocast",
            "1",
            "--scale_mode",
            "window",
        ]
    elif model == "astronex":
        script = L / "astronex_run.py"
        args = [
            "--out",
            o,
            "--arms",
            str(w / L / "astronex_arms.json"),
            "--arm",
            corrected,
            "--videos",
            scene,
            "--seed",
            "7",
            "--scale",
            "1",
        ]
    elif model in ["memorizon", "worldplay"]:
        script = F / ("mem_fair.py" if model == "memorizon" else "wp_fair.py")
        args = [
            "--out",
            o,
            "--arms",
            str(w / A / "baseline_loop_20261028/arms.json"),
            "--arm",
            corrected,
            "--scenes",
            scene,
            "--seeds",
            "7",
        ]
    else:
        script = N / ("lingbot_pair.py" if model == "lingbot" else "dreamx_run.py")
        args = (
            common
            + ["--seed", "7"]
            + (
                ["--arms", arm, "--frames", "173"]
                if model == "lingbot"
                else ["--arm", arm, "--latents", "42"]
            )
        )
    if model == "worldplay" and (Path(b) / "index.json").is_file():
        row = next(
            r
            for r in json.loads((Path(b) / "index.json").read_text())["scenes"]
            if r["name"] == scene
        )
        if "native_units_per_anchor_unit" in row:
            args += ["--unit", str(row["native_units_per_anchor_unit"])]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        LONG_BATCH=b,
        CAMERA_BATCH=b,
        GV_SCALE_CACHE=str(w / "camera_loop_calibration"),
        PYTHONUNBUFFERED="1",
    )
    return [str(python), str(w / script), *args], env


def doctor(workspace, models=None, python_overrides=None, conda_root=None):
    w = Path(workspace).resolve()
    found = []
    overrides = python_overrides or {}
    conda = Path(conda_root or Path.home() / "miniconda3")
    for name in models or profiles():
        p = profile(name)
        if "repo" not in p:
            found.append(dict(model=name, status=p["status"], note=p["notes"]))
            continue
        interpreter = overrides.get(
            name, p["python"].replace("@CONDA_ROOT@", str(conda))
        )
        interpreter = Path(interpreter)
        interpreter = interpreter if interpreter.is_absolute() else w / interpreter
        path = w / p["repo"]["path"]
        commit = None
        if (path / ".git").exists():
            r = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            commit = r.stdout.strip() if r.returncode == 0 else None
        found.append(
            dict(
                model=name,
                status=p["status"],
                interpreter=str(interpreter),
                interpreter_exists=interpreter.is_file(),
                repo_exists=path.is_dir(),
                expected_commit=p["repo"]["commit"],
                actual_commit=commit,
                commit_matches=commit == p["repo"]["commit"],
                note=p["notes"],
            )
        )
    return found
