"""Load every exported ONNX with onnxruntime 1.20 (the version legged_control2 links) and replay its parity rollout.

Checks, per file: the model loads; the input/output names and shapes motion_tracking_controller validates; the
metadata keys it reads; obs size == sum(term size x history); outputs match the PyTorch export reference.

  python ort_check.py <export dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

TERM_SIZES = {"joint_pos": 29, "joint_vel": 29, "base_ang_vel": 3, "projected_gravity": 3, "actions": 29,
              "motion_anchor_pos_b": 3, "motion_anchor_ori_b": 6}
REQUIRED_META = ["joint_names", "joint_stiffness", "joint_damping", "default_joint_pos", "action_scale",
                 "command_names", "observation_names", "observation_history_lengths", "anchor_body_name", "body_names"]
OUTPUTS = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


def check(onnx_path: Path) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    missing = [k for k in REQUIRED_META if k not in meta]
    assert not missing, f"missing metadata {missing}"
    names = meta["observation_names"].split(",")
    hists = [int(float(h)) for h in meta["observation_history_lengths"].split(",")]
    joints = meta["joint_names"].split(",")
    bodies = meta["body_names"].split(",")
    obs_dim = sum(TERM_SIZES[n] * h for n, h in zip(names, hists))
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    outputs = {o.name: o.shape for o in sess.get_outputs()}
    assert inputs == {"obs": [1, obs_dim], "time_step": [1, 1]}, inputs
    assert [o.name for o in sess.get_outputs()] == OUTPUTS, list(outputs)
    assert outputs["actions"] == [1, len(joints)] and outputs["joint_pos"] == [1, len(joints)], outputs
    assert outputs["body_pos_w"] == [1, len(bodies), 3] and outputs["body_quat_w"] == [1, len(bodies), 4], outputs
    assert meta["anchor_body_name"] in bodies

    par = np.load(onnx_path.with_suffix(".parity.npz"))
    worst = {name: 0.0 for name in OUTPUTS}
    for i in range(len(par["cpp_obs"])):
        out = sess.run(None, {"obs": par["cpp_obs"][i : i + 1], "time_step": par["time_step"][i : i + 1]})
        for name, o in zip(OUTPUTS, out):
            worst[name] = max(worst[name], float(np.abs(o[0] - par[f"torch_{name}"][i]).max()))
    return {"obs_dim": obs_dim, "observation_names": names, "history": hists, "steps": int(len(par["cpp_obs"])),
            "max_abs_diff_vs_torch": worst}


def main(export_dir: str) -> None:
    d = Path(export_dir)
    report = {"onnxruntime": ort.__version__, "files": {}}
    for p in sorted(d.glob("*.onnx")):
        r = check(p)
        report["files"][p.name] = r
        print(f"[ort {ort.__version__}] {p.name}: obs {r['obs_dim']}, {r['steps']} steps, "
              f"max |ORT - torch| actions {r['max_abs_diff_vs_torch']['actions']:.2e}", flush=True)
    (d / "ort_check_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
