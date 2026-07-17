#preprocessing_libero_xvla.py
#!/usr/bin/env python3
# Parallel LIBERO -> X-VLA preprocessing.
#
# Writes, per demo, ONE .hdf5 in the format LiberoHandler expects:
#   abs_action_6d : float32 [T, 10]  = xyz(3) + rot6d(6) + gripper_raw(1)
#   agentview_rgb : uint8   [T+1,H,W,3]   (handler drops frame 0)
#   eye_in_hand_rgb: uint8  [T+1,H,W,3]
#   language_instruction : scalar utf-8 string
#
# Parallelism: each worker process handles ONE task at a time and builds its
# OWN OffScreenRenderEnv (its own EGL context). MuJoCo/EGL contexts cannot be
# shared or forked across processes, so we use the 'spawn' start method and
# never create an env in the parent.
#
# Run in the `libero` env:
#   MUJOCO_GL=egl python preprocess_libero_xvla_parallel.py \
#       --raw_data_root /data2/daniel/Documents/VLAReplica2/libero/datasets \
#       --out_dir       /data2/daniel/libero/libero_h5 \
#       --suites libero_10 \
#       --workers 8

import os
import json
import argparse
import multiprocessing as mp

import numpy as np
import h5py
from tqdm import tqdm


# ---- 6D rotation: match libero_client.py Mat_to_Rotate6D (first two columns) ----
def axisangle_to_rotate6d(aa, T):
    R = T.quat2mat(T.axisangle2quat(aa))   # (3,3)
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)


def _process_one_task(job):
    """Runs inside a worker process. Builds its own env, processes all demos
    for a single (suite, task_id). Returns list of written file paths.

    Heavy imports happen HERE, not at module top level, so the parent never
    initializes libero/robosuite/EGL before the fork/spawn.
    """
    (suite_name, task_id, raw_suite_dir, suite_out,
     demos_per_task, skip_failed, cam_h, cam_w, egl_device) = job

    # Each process gets its own renderer; pin to a GPU if requested.
    os.environ.setdefault("MUJOCO_GL", "egl")
    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(egl_device)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.transform_utils as T

    bench = benchmark.get_benchmark_dict()
    task_suite = bench[suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    # Build the env ONCE for this task; reuse across its demos.
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file, camera_heights=cam_h, camera_widths=cam_w
    )
    env.seed(0)

    raw_path = os.path.join(raw_suite_dir, f"{task.name}_demo.hdf5")
    written = []

    try:
        with h5py.File(raw_path, "r") as raw:
            demo_keys = [k for k in raw["data"].keys() if k.startswith("demo_")]
            # Honor demos_per_task as a cap.
            demo_keys = sorted(demo_keys, key=lambda s: int(s.split("_")[1]))
            if demos_per_task is not None:
                demo_keys = demo_keys[:demos_per_task]

            for dkey in demo_keys:
                demo_id = int(dkey.split("_")[1])
                orig_states = raw["data"][dkey]["states"][()]
                actions = raw["data"][dkey]["actions"][()]   # [T,7] relative

                env.reset()
                env.set_init_state(orig_states[0])
                for _ in range(10):  # settle
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))

                agent_imgs = [np.flip(np.flip(obs["agentview_image"], 0), 1)]
                wrist_imgs = [obs["robot0_eye_in_hand_image"]]
                abs6d = []
                done = False

                for action in actions:
                    obs, reward, done, info = env.step(action)
                    goal_pos = env.env.robots[0].controller.goal_pos
                    goal_ori_mat = env.env.robots[0].controller.goal_ori
                    aa = T.quat2axisangle(T.mat2quat(goal_ori_mat))
                    rot6d = axisangle_to_rotate6d(aa, T)
                    abs6d.append(np.concatenate([goal_pos, rot6d, action[-1:]]))
                    agent_imgs.append(np.flip(np.flip(obs["agentview_image"], 0), 1))
                    wrist_imgs.append(obs["robot0_eye_in_hand_image"])

                if skip_failed and not done:
                    continue

                abs6d = np.stack(abs6d).astype(np.float32)
                agentview = np.stack(agent_imgs).astype(np.uint8)
                wrist = np.stack(wrist_imgs).astype(np.uint8)

                out_path = os.path.join(suite_out, f"{task.name}_demo_{demo_id}.hdf5")
                with h5py.File(out_path, "w") as f:
                    f.create_dataset("abs_action_6d", data=abs6d, compression="gzip")
                    f.create_dataset("agentview_rgb", data=agentview, compression="gzip")
                    f.create_dataset("eye_in_hand_rgb", data=wrist, compression="gzip")
                    f.create_dataset("language_instruction", data=np.bytes_(str(task.language)))
                written.append(out_path)
    finally:
        env.close()

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_data_root", required=True,
                    help="Dir holding <suite>/<task>_demo.hdf5 (raw LIBERO).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--suites", nargs="+", default=["libero_10"])
    ap.add_argument("--demos_per_task", type=int, default=50)
    ap.add_argument("--no_skip_failed", action="store_true",
                    help="Keep demos even if replay does not reach success.")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2),
                    help="Number of parallel worker processes.")
    ap.add_argument("--cam_h", type=int, default=256)
    ap.add_argument("--cam_w", type=int, default=256)
    ap.add_argument("--egl_device", type=int, default=None,
                    help="GPU index for EGL rendering (sets MUJOCO_EGL_DEVICE_ID).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the job list in the PARENT without importing libero/EGL.
    # We need task counts per suite; do that import in a short-lived child to
    # keep the parent EGL-clean. Simpler: import here is fine because the parent
    # never creates an OffScreenRenderEnv. benchmark.get_benchmark_dict() does
    # not touch the GL context.
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()

    jobs = []
    for suite_name in args.suites:
        task_suite = bench[suite_name]()
        raw_suite_dir = os.path.join(args.raw_data_root, suite_name)
        suite_out = os.path.join(args.out_dir, suite_name)
        os.makedirs(suite_out, exist_ok=True)
        for task_id in range(len(task_suite.tasks)):
            jobs.append((
                suite_name, task_id, raw_suite_dir, suite_out,
                args.demos_per_task, not args.no_skip_failed,
                args.cam_h, args.cam_w, args.egl_device,
            ))

    print(f"Dispatching {len(jobs)} tasks across {args.workers} workers...")

    ctx = mp.get_context("spawn")  # never fork an EGL/CUDA-touched process
    datalist = []
    with ctx.Pool(processes=args.workers) as pool:
        for written in tqdm(pool.imap_unordered(_process_one_task, jobs),
                            total=len(jobs), desc="tasks"):
            datalist.extend(written)

    datalist.sort()
    meta = {
        "dataset_name": "libero",
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
        "datalist": datalist,
    }
    meta_path = os.path.join(args.out_dir, "libero_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {len(datalist)} demos. Meta: {meta_path}")


if __name__ == "__main__":
    # 'spawn' is required: forking after EGL/CUDA init corrupts the GL context.
    mp.set_start_method("spawn", force=True)
    main()
