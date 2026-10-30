import argparse, json, subprocess
from dataclasses import asdict
from pathlib import Path
from .registry import profiles
from .integrations import materialize, doctor, command
from .alignment import align_model
from .datasets import import_batch, validate_batch, fingerprint
from .experiments import plan_experiment, execute_plan, check_lyra_cache


def main():
    p = argparse.ArgumentParser(description="Model-aware closed-loop camera correction")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "models", help="List available native adapters and validation limits"
    )
    a = sub.add_parser("prepare", help="Materialize into an empty workspace")
    a.add_argument("--workspace", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--clone", action="store_true")
    a = sub.add_parser(
        "doctor",
        help="Inspect local repositories and interpreters without loading weights",
    )
    a.add_argument("--workspace", required=True)
    a.add_argument("--config")
    a.add_argument("--model", choices=list(profiles()))
    a = sub.add_parser(
        "run", help="Print one native adapter command; execution is explicit"
    )
    for key in ["workspace", "model", "python", "batch", "scene", "out"]:
        a.add_argument("--" + key, required=True)
    a.add_argument(
        "--arm",
        default="correction",
        choices=["baseline", "native", "sham", "correction", "replan_measured"],
    )
    a.add_argument("--device", default="0")
    a.add_argument("--quality", choices=["report", "enforce"], default="enforce")
    a.add_argument("--execute", action="store_true")
    a.add_argument("--t5-cache", help="Exact-caption cache for this Lyra batch")
    a = sub.add_parser(
        "import-batch",
        help="Convert supplied registered geometry into the legacy batch format",
    )
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument(
        "--resize",
        action="store_true",
        help="Explicitly resize registered RGB/depth/K to 864x480",
    )
    a = sub.add_parser(
        "validate-batch",
        help="CPU checks for dataset geometry and a model's default schedule",
    )
    a.add_argument("--batch", required=True)
    a.add_argument("--model", choices=list(profiles()))
    a.add_argument("--scenes", nargs="+")
    a = sub.add_parser(
        "run-batch", help="Plan or serially execute selected scenes and matched arms"
    )
    for key in ["workspace", "model", "python", "batch", "out"]:
        a.add_argument("--" + key, required=True)
    a.add_argument(
        "--scenes", nargs="+", help="Omit to select all scenes in index.json"
    )
    a.add_argument("--arms", nargs="+", default=["baseline", "correction"])
    a.add_argument("--device", default="0")
    a.add_argument("--quality", choices=["report", "enforce"], default="enforce")
    a.add_argument("--t5-cache")
    a.add_argument("--execute", action="store_true")
    a = sub.add_parser(
        "align", help="Fit native units from registered depth arrays or native metadata"
    )
    a.add_argument("--model", required=True)
    a.add_argument(
        "--depths", help="NPZ containing registered reference_depth and native_depth"
    )
    a.add_argument("--metric-scale", type=float)
    a.add_argument("--native-scale", type=float)
    a.add_argument("--evidence", default="")
    a.add_argument("--out", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--anchor-id", required=True)
    a = p.parse_args()
    try:
        if a.cmd == "models":
            result = profiles()
        elif a.cmd == "import-batch":
            result = import_batch(a.manifest, a.out, a.resize)
        elif a.cmd == "validate-batch":
            result = validate_batch(a.batch, a.model, a.scenes)
            if not result["passed"]:
                print(json.dumps(result, indent=2))
                p.exit(2)
        elif a.cmd == "run-batch":
            result = plan_experiment(
                a.workspace,
                a.model,
                a.python,
                a.batch,
                a.out,
                a.scenes,
                a.arms,
                a.device,
                a.quality,
                a.t5_cache,
            )
            if a.execute:
                print(json.dumps(result, indent=2), flush=True)
                result = execute_plan(result)
        elif a.cmd == "prepare":
            result = materialize(
                a.workspace, json.loads(Path(a.config).read_text()), a.clone
            )
        elif a.cmd == "doctor":
            config = json.loads(Path(a.config).read_text()) if a.config else {}
            result = doctor(
                a.workspace,
                [a.model] if a.model else None,
                config.get("python"),
                config.get("conda_root"),
            )
        elif a.cmd == "align":
            import numpy as np

            kw = {}
            if a.depths:
                with np.load(a.depths) as z:
                    kw = {
                        "reference_depth": z["reference_depth"],
                        "native_depth": z["native_depth"],
                        "mask": z["mask"] if "mask" in z else None,
                    }
            alignment, stats = align_model(
                a.model,
                **kw,
                metres_per_reference_unit=a.metric_scale,
                native_units_per_reference_unit=a.native_scale,
                evidence=a.evidence,
            )
            alignment = alignment.save(
                a.out,
                anchor_fingerprint=stats.get("reference_sha256", a.anchor_id),
                checkpoint=a.checkpoint,
                context=a.anchor_id,
            )
            result = dict(alignment=asdict(alignment), diagnostics=stats, receipt=a.out)
        else:
            report = validate_batch(a.batch, a.model, [a.scene])
            if not report["passed"]:
                raise ValueError(json.dumps(report, indent=2))
            argv, env = command(
                a.workspace,
                a.model,
                a.python,
                a.batch,
                a.scene,
                a.out,
                a.arm,
                a.device,
                a.quality,
            )
            env["GV_SCALE_CACHE"] = str(
                Path(a.workspace).resolve()
                / "camera_loop_calibration"
                / fingerprint(a.batch)
            )
            if a.t5_cache:
                env["CAMERA_LYRA_T5_CACHE"] = str(Path(a.t5_cache).resolve())
            result = {
                "argv": argv,
                "environment": {
                    k: env[k]
                    for k in [
                        "CUDA_VISIBLE_DEVICES",
                        "LONG_BATCH",
                        "CAMERA_BATCH",
                        "GV_SCALE_CACHE",
                    ]
                },
                "execute": a.execute,
                "batch_validation": report,
                "t5_cache": env.get("CAMERA_LYRA_T5_CACHE"),
            }
            if a.execute:
                if Path(a.out).exists():
                    raise ValueError("Output exists; use a fresh directory")
                if not Path(argv[0]).is_file() or not Path(argv[1]).is_file():
                    raise ValueError(
                        "Interpreter or adapter missing; run doctor/prepare first"
                    )
                print(json.dumps(result, indent=2), flush=True)
                if a.model == "lyra":
                    env["CAMERA_LYRA_T5_CACHE"] = check_lyra_cache(
                        a.workspace, a.batch, [a.scene], a.t5_cache
                    )
                subprocess.run(argv, env=env, check=True)
                return
        print(json.dumps(result, indent=2))
    except (
        ValueError,
        OSError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as e:
        p.exit(2, str(e) + "\n")


if __name__ == "__main__":
    main()
