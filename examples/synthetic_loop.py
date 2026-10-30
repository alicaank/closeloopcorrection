"""CPU-only integration demo; simulated under-execution, not a model evaluation."""

import numpy as np
from camera_loop import CameraAlignment, CorrectionSession, Measurement


def run(correct):
    alignment = CameraAlignment(3.0, evidence="synthetic native unit contract")
    controller = CorrectionSession(alignment)
    plan = np.repeat(np.eye(4)[None], 41, axis=0)
    plan[:, 0, 3] = np.linspace(0, 1, 41)
    actual = np.eye(4)
    previous = np.eye(4)
    for lo in range(0, 40, 10):
        future = plan[lo + 1 : lo + 11]
        if correct and lo:
            cmd, event = controller.correct(
                Measurement(
                    actual.copy(),
                    lo,
                    {"passed": True, "source": "synthetic exact pose"},
                ),
                planned_endpoint=plan[lo],
                future_reference_c2w=future,
                future_frame_ids=np.arange(lo + 1, lo + 11),
                available_last_frame=lo,
                previous_native_command=previous,
            )
        else:
            cmd = alignment.encode(future)
        for target in cmd:
            actual[:3, 3] += 0.5 * (target[:3, 3] - previous[:3, 3]) / 3
            previous = target.copy()
    return float(np.linalg.norm(actual[:3, 3] - plan[-1, :3, 3]))


if __name__ == "__main__":
    b, c = run(False), run(True)
    assert c < b
    print(
        {
            "simulation_only": True,
            "baseline_endpoint_error": b,
            "corrected_endpoint_error": c,
        }
    )
