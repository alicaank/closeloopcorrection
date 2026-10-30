"""Validated, serial scene/arm execution. No implicit resume, scheduling or GPU probing."""

import ast
import json
import os
import subprocess
import time
from pathlib import Path

from .datasets import validate_batch, index_rows, fingerprint
from .integrations import command, C, V


def plan_experiment(
    workspace,
    model,
    python,
    batch,
    out,
    scenes=None,
    arms=("baseline", "correction"),
    device="0",
    quality="enforce",
    t5_cache=None,
):
    root = Path(out).resolve()
    if root.exists():
        raise ValueError("Experiment output exists; use a fresh directory")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("Choose unique nonempty arms")
    report = validate_batch(batch, model, scenes)
    if not report["passed"]:
        raise ValueError(json.dumps(report, indent=2))
    _, rows = index_rows(batch, scenes)
    digest = fingerprint(batch)
    jobs = []
    for row in rows:
        for arm in arms:
            destination = root / row["name"] / arm
            argv, env = command(
                workspace,
                model,
                python,
                batch,
                row["name"],
                destination,
                arm,
                device,
                quality,
            )
            jobs.append(
                dict(scene=row["name"], arm=arm, out=str(destination), argv=argv)
            )
    return dict(
        version=1,
        model=model,
        workspace=str(Path(workspace).resolve()),
        batch=str(Path(batch).resolve()),
        batch_sha256=digest,
        out=str(root),
        device=device,
        quality=quality,
        t5_cache=str(Path(t5_cache).resolve()) if t5_cache else None,
        calibration_cache=str(
            Path(workspace).resolve() / "camera_loop_calibration" / digest
        ),
        validation=report,
        jobs=jobs,
        seed=7,
        note="Data preflight only. Environment/weights/GPU qualification are separate. Native schedules and seed 7 are retained.",
    )


def check_lyra_cache(workspace, batch, scenes, cache):
    """Inspect tensor cache in the native child interpreter without using CUDA."""
    cache = Path(cache) if cache else Path(workspace) / C / "lyra_t5.pt"
    if not cache.is_file():
        raise ValueError(
            "Missing Lyra caption cache; precompute this batch and pass --t5-cache: "
            + str(cache)
        )
    source = Path(workspace) / V / "snapshots/lyra_run.py"
    native = None
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PY" for t in node.targets
        ):
            native = ast.literal_eval(node.value)
    if not native or not Path(native).is_file():
        raise ValueError("Lyra child interpreter is missing; map it during prepare")
    _, rows = index_rows(batch, scenes)
    wanted = set()
    for r in rows:
        wanted.update(
            json.loads(
                (Path(batch) / "captions" / f'{r["name"]}.json').read_text()
            ).values()
        )
    script = """import json,sys,torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
wanted=json.load(sys.stdin)
prompts=c.get('prompts',{c.get('prompt'):c.get('embedding')})
missing=[s for s in wanted if s not in prompts]
invalid=[s for s in wanted if s in prompts and (not isinstance(prompts[s],torch.Tensor) or prompts[s].ndim!=2 or prompts[s].shape[0]<512 or prompts[s].shape[1]!=4096)]
print(json.dumps({'missing_caption_count':len(missing),'invalid_embedding_count':len(invalid)}))
sys.exit(2 if missing or invalid else 0)
"""
    r = subprocess.run(
        [native, "-c", script, str(cache)],
        input=json.dumps(sorted(wanted)),
        text=True,
        capture_output=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    if r.returncode:
        raise ValueError(
            "Lyra cache preflight failed: " + r.stdout.strip() + " " + r.stderr[-1000:]
        )
    return str(cache.resolve())


def execute_plan(plan):
    root = Path(plan["out"])
    if root.exists():
        raise ValueError("Experiment output exists; no implicit resume or overwrite")
    if fingerprint(plan["batch"]) != plan["batch_sha256"]:
        raise ValueError("Dataset changed since planning")
    for job in plan["jobs"]:
        if not Path(job["argv"][0]).is_file() or not Path(job["argv"][1]).is_file():
            raise ValueError(
                "Missing native interpreter or adapter; run doctor and prepare first"
            )
    cache = plan.get("t5_cache")
    if plan["model"] == "lyra":
        cache = check_lyra_cache(
            plan["workspace"],
            plan["batch"],
            list(dict.fromkeys(j["scene"] for j in plan["jobs"])),
            cache,
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    state = dict(status="running", jobs=[])

    def save():
        temp = root / "status.json.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(root / "status.json")

    save()
    for job in plan["jobs"]:
        _, env = command(
            plan["workspace"],
            plan["model"],
            job["argv"][0],
            plan["batch"],
            job["scene"],
            job["out"],
            job["arm"],
            plan["device"],
            plan["quality"],
        )
        env["GV_SCALE_CACHE"] = plan["calibration_cache"]
        if cache:
            env["CAMERA_LYRA_T5_CACHE"] = cache
        log = root / f'{job["scene"]}--{job["arm"]}.log'
        record = dict(
            scene=job["scene"],
            arm=job["arm"],
            started=time.time(),
            status="running",
            log=str(log),
        )
        state["jobs"].append(record)
        save()
        try:
            with log.open("x") as f:
                r = subprocess.run(
                    job["argv"], env=env, stdout=f, stderr=subprocess.STDOUT
                )
            record.update(
                returncode=r.returncode,
                finished=time.time(),
                status="process_succeeded" if r.returncode == 0 else "failed",
            )
            if r.returncode:
                state["status"] = "failed"
                save()
                raise RuntimeError(f"Job failed; stopped remaining jobs. See {log}")
        except (OSError, KeyboardInterrupt) as e:
            record.update(status="interrupted_or_error", error=str(e))
            state["status"] = "failed"
            save()
            raise
        save()
    state["status"] = "processes_succeeded"
    state["note"] = (
        "Exit codes recorded; inspect native output receipts and independent evaluation before accepting results."
    )
    save()
    return state
