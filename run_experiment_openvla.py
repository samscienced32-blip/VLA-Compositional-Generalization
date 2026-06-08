"""
VLA COMPOSITIONALITY EXPERIMENT — Standard OpenVLA
====================================================
Fallback version using openvla/openvla-7b-finetuned-libero-spatial.
Single camera, single-action-per-step, no action head / proprio needed.

Usage:
  MUJOCO_GL=egl python run_experiment_openvla.py
  MUJOCO_GL=egl python run_experiment_openvla.py --seed 1
  MUJOCO_GL=egl python run_experiment_openvla.py --visualize --num_tasks 1 --rollouts 2
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

# ── ARGS ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--visualize",     action="store_true")
parser.add_argument("--num_tasks",     type=int,   default=10)
parser.add_argument("--rollouts",      type=int,   default=10)
parser.add_argument("--perturb_noise", type=float, default=0.05)
parser.add_argument("--cells",         type=str,   default="1,2,3,4")
parser.add_argument("--seed",          type=int,   default=0)
args = parser.parse_args()

VISUALIZE    = args.visualize
NUM_TASKS    = args.num_tasks
N_ROLLOUTS   = args.rollouts
NOISE        = args.perturb_noise
CELLS_TO_RUN = [int(c) for c in args.cells.split(",")]
SEED         = args.seed
SEED_SUFFIX  = "" if SEED == 0 else f"_seed{SEED}"

os.environ.setdefault("MUJOCO_GL", "egl")

# ── PATHS ──────────────────────────────────────────────────────
LIBERO_PATH  = os.path.expanduser("~/LIBERO")
OPENVLA_PATH = os.path.expanduser("~/openvla-oft")   # still need libero_utils
RESULTS_DIR  = os.path.expanduser("~/VLMmodel/results")
FIGURES_DIR  = os.path.expanduser("~/VLMmodel/figures")
PARA_FILE    = os.path.expanduser("~/VLMmodel/paraphrases.json")

sys.path.insert(0, LIBERO_PATH)
sys.path.insert(0, OPENVLA_PATH)
sys.path.insert(0, os.path.join(OPENVLA_PATH, "experiments", "robot"))
sys.path.insert(0, os.path.join(OPENVLA_PATH, "experiments", "robot", "libero"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero_utils import quat2axisangle
from robot_utils import normalize_gripper_action, invert_gripper_action

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model_id"         : "openvla/openvla-7b-finetuned-libero-spatial",
    "task_suite_name"  : "libero_spatial",
    "num_tasks"        : NUM_TASKS,
    "rollouts_per_task": N_ROLLOUTS,
    "max_steps"        : 300,          # standard OpenVLA runs single-step, allow more steps
    "num_steps_wait"   : 10,
    "img_size"         : 256,
    "load_in_8bit"     : True,
}

DEVICE = torch.device("cuda:0")

# ── LOAD STANDARD OPENVLA ──────────────────────────────────────
print("=" * 60)
print("Loading standard OpenVLA (openvla-7b-finetuned-libero-spatial)...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    CONFIG["model_id"],
    torch_dtype=torch.bfloat16,
    load_in_8bit=CONFIG["load_in_8bit"],
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map={"": 0},
).eval()
print(f"✓ Loaded  ({time.time()-t0:.1f}s)  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Resolve unnorm key
_stats = model.norm_stats if hasattr(model, "norm_stats") else {}
UNNORM_KEY = CONFIG["task_suite_name"]
if UNNORM_KEY not in _stats:
    for candidate in [f"{UNNORM_KEY}_no_noops", "libero_spatial", "libero_spatial_no_noops"]:
        if candidate in _stats:
            UNNORM_KEY = candidate
            break
print(f"✓ unnorm_key = '{UNNORM_KEY}'")

# ── LOAD TASKS ─────────────────────────────────────────────────
benchmark_dict = benchmark.get_benchmark_dict()
task_suite     = benchmark_dict[CONFIG["task_suite_name"]]()
tasks          = task_suite.tasks[:NUM_TASKS]
task_descriptions = [t.language for t in tasks]

print(f"\nLIBERO-Spatial tasks ({NUM_TASKS} of 10):")
for i, d in enumerate(task_descriptions):
    print(f"  [{i:02d}] {d}")

# ── PARAPHRASES ────────────────────────────────────────────────
if os.path.exists(PARA_FILE):
    with open(PARA_FILE) as f:
        para_data = json.load(f)
    PARAPHRASES = {}
    for item in para_data:
        idx = item["task_idx"]
        if idx >= NUM_TASKS:
            continue
        item["original"] = task_descriptions[idx]
        PARAPHRASES[idx] = item
    print(f"\n✓ Paraphrases loaded from {PARA_FILE}")
else:
    PARAPHRASES = {
        i: {"task_idx": i, "original": task_descriptions[i],
            "variants": [task_descriptions[i]] * 3}
        for i in range(NUM_TASKS)
    }
PARA_TYPE_LABELS = {0: "synonym_swap", 1: "restructure", 2: "colloquial"}

# ── INFERENCE ──────────────────────────────────────────────────
def get_action(image_np, instruction):
    """Single-step action from standard OpenVLA. No chunking, no proprio."""
    image = Image.fromarray(image_np)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    action_np = action.detach().cpu().float().numpy() if hasattr(action, "detach") else np.array(action)
    if action_np.ndim == 2:
        action_np = action_np[0]   # [1,7] → [7]
    action_np = normalize_gripper_action(action_np, binarize=True)
    action_np = invert_gripper_action(action_np)
    return action_np

# ── ENV FACTORY ────────────────────────────────────────────────
def make_env(task, perturb=False):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    kwargs = {
        "bddl_file_name": bddl,
        "camera_heights": CONFIG["img_size"],
        "camera_widths" : CONFIG["img_size"],
        "camera_names"  : ["agentview"],
        "has_renderer"  : False, "has_offscreen_renderer": True,
        "ignore_done"   : False, "use_camera_obs": True,
        "reward_shaping": False, "control_freq": 20,
    }
    env = OffScreenRenderEnv(**kwargs)
    env.seed(SEED)
    return env

# ── EPISODE RUNNER ─────────────────────────────────────────────
def run_episode(env, instruction, initial_state, perturb=False, save_video_path=None):
    env.reset()
    obs = env.set_init_state(initial_state)
    if perturb:
        try:
            sim = env.env.sim
            sim.data.qpos[:7] += np.random.normal(0, NOISE, 7)
            sim.forward()
            obs = env.env._get_observations()
        except Exception as e:
            print(f"  [perturb warning] {e}")

    frames = [] if save_video_path else None

    for _ in range(CONFIG["num_steps_wait"]):
        obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    for step in range(CONFIG["max_steps"]):
        img = obs["agentview_image"][::-1, ::-1].copy()
        action = get_action(img, instruction)

        obs, reward, done, _ = env.step(action.tolist())
        task_success = bool(done)

        if reward != 0 or task_success:
            print(f"  *** step {step+1}: reward={reward:.3f} done={done} ***")

        if frames is not None:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())

        if VISUALIZE:
            disp = cv2.cvtColor(obs["agentview_image"][::-1, ::-1], cv2.COLOR_RGB2BGR)
            status = "SUCCESS" if task_success else f"step {step+1}"
            cv2.putText(disp, f"{status} | {instruction[:55]}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if task_success else (255, 255, 255), 1)
            cv2.imshow("LIBERO — standard OpenVLA", disp)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

        if task_success or done:
            if frames is not None:
                import imageio
                imageio.mimsave(save_video_path, frames, fps=20)
            return True, step + 1

    # last-chance success check
    try:
        if bool(env.check_success()):
            return True, CONFIG["max_steps"]
    except Exception:
        pass

    if frames is not None:
        import imageio
        imageio.mimsave(save_video_path, frames, fps=20)
    return False, CONFIG["max_steps"]

def save(data, fname):
    path = f"{RESULTS_DIR}/{fname}"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved → {path}")

# ── CELL RUNNERS ───────────────────────────────────────────────
def run_cell(cell_name, tasks_list, instruction_fn, perturb, save_video=False):
    results = []
    for task_idx, task in enumerate(tasks_list):
        instruction    = instruction_fn(task_idx)
        env            = make_env(task, perturb=perturb)
        initial_states = task_suite.get_task_init_states(task_idx)
        successes, steps_list = [], []
        print(f"\n[{cell_name}] Task {task_idx:02d}: {instruction[:65]}...")
        for r in range(CONFIG["rollouts_per_task"]):
            init_state = initial_states[r % len(initial_states)]
            t_ep = time.time()
            vid_path = (f"{FIGURES_DIR}/{cell_name}_task{task_idx:02d}_r{r:02d}.gif"
                        if save_video else None)
            success, steps = run_episode(env, instruction, init_state,
                                         perturb=perturb,
                                         save_video_path=vid_path)
            successes.append(int(success))
            steps_list.append(steps)
            print(f"  Rollout {r+1:02d}: {'✓' if success else '✗'} "
                  f"({steps} steps, {time.time()-t_ep:.1f}s)")
        env.close()
        sr = float(np.mean(successes))
        results.append({
            "cell": cell_name, "task_idx": task_idx,
            "instruction": instruction,
            "successes": successes, "steps": steps_list,
            "success_rate": sr, "n_rollouts": len(successes),
        })
        print(f"  → SR: {sr:.2f}  ({sum(successes)}/{len(successes)})")
    return results


def run_cell2(tasks_list):
    results = []
    for task_idx, task in enumerate(tasks_list):
        para           = PARAPHRASES[task_idx]
        env            = make_env(task, perturb=False)
        initial_states = task_suite.get_task_init_states(task_idx)
        print(f"\n[C2_linguistic] Task {task_idx:02d}: {para['original'][:60]}...")
        for v_idx, variant in enumerate(para["variants"]):
            successes, steps_list = [], []
            para_type = PARA_TYPE_LABELS.get(v_idx, f"variant_{v_idx}")
            print(f"  [{para_type}]: {variant[:60]}...")
            for r in range(CONFIG["rollouts_per_task"]):
                init_state = initial_states[r % len(initial_states)]
                t_ep = time.time()
                success, steps = run_episode(env, variant, init_state)
                successes.append(int(success))
                steps_list.append(steps)
                print(f"    Rollout {r+1:02d}: {'✓' if success else '✗'} "
                      f"({steps} steps, {time.time()-t_ep:.1f}s)")
            sr = float(np.mean(successes))
            results.append({
                "cell": "C2_linguistic", "task_idx": task_idx,
                "variant_idx": v_idx, "para_type": para_type,
                "original": para["original"], "instruction": variant,
                "successes": successes, "steps": steps_list,
                "success_rate": sr, "n_rollouts": len(successes),
            })
            print(f"    → SR: {sr:.2f}  ({sum(successes)}/{len(successes)})")
        env.close()
    return results

# ── RUN ────────────────────────────────────────────────────────
total_start = time.time()

if VISUALIZE:
    print("\n" + "="*60 + "\nSANITY CHECK (1 task × 2 rollouts)\n" + "="*60)
    run_cell("C1_sanity", tasks[:1], lambda i: task_descriptions[i],
             perturb=False, save_video=True)
    cv2.destroyAllWindows()
    sys.exit(0)

c1_mean = c2_mean = c3_mean = c4_mean = 0.0

if 1 in CELLS_TO_RUN:
    print("\n" + "="*60 + "\nCELL 1: BASELINE\n" + "="*60)
    c1 = run_cell("C1_baseline", tasks, lambda i: task_descriptions[i], perturb=False)
    save(c1, f"cell1_baseline{SEED_SUFFIX}.json")
    c1_mean = float(np.mean([r["success_rate"] for r in c1]))
    print(f"\n→ C1 Mean SR: {c1_mean:.3f}")

if 2 in CELLS_TO_RUN:
    print("\n" + "="*60 + "\nCELL 2: LINGUISTIC STRESS\n" + "="*60)
    c2 = run_cell2(tasks)
    save(c2, f"cell2_linguistic{SEED_SUFFIX}.json")
    c2_mean = float(np.mean([r["success_rate"] for r in c2]))
    print(f"\n→ C2 Mean SR: {c2_mean:.3f}")

if 3 in CELLS_TO_RUN:
    print("\n" + "="*60 + "\nCELL 3: VISUAL STRESS\n" + "="*60)
    c3 = run_cell("C3_visual", tasks, lambda i: task_descriptions[i], perturb=True)
    save(c3, f"cell3_visual{SEED_SUFFIX}.json")
    c3_mean = float(np.mean([r["success_rate"] for r in c3]))
    print(f"\n→ C3 Mean SR: {c3_mean:.3f}")

if 4 in CELLS_TO_RUN:
    print("\n" + "="*60 + "\nCELL 4: FULL NOVEL\n" + "="*60)
    c4 = run_cell("C4_combined", tasks,
                  lambda i: PARAPHRASES[i]["variants"][0], perturb=True)
    save(c4, f"cell4_combined{SEED_SUFFIX}.json")
    c4_mean = float(np.mean([r["success_rate"] for r in c4]))
    print(f"\n→ C4 Mean SR: {c4_mean:.3f}")

if all(c in CELLS_TO_RUN for c in [1, 2, 3, 4]):
    elapsed = time.time() - total_start
    print("\n" + "="*60 + "\nQUICK CGG PREVIEW\n" + "="*60)
    print(f"  C1={c1_mean:.3f}  C2={c2_mean:.3f}  C3={c3_mean:.3f}  C4={c4_mean:.3f}")
    print(f"  CGG_linguistic  = {c1_mean - c2_mean:+.3f}")
    print(f"  CGG_visual      = {c1_mean - c3_mean:+.3f}")
    print(f"  CGG_full        = {c1_mean - c4_mean:+.3f}")
    cgg_syn = (c1_mean - c4_mean) - (c1_mean - c2_mean) - (c1_mean - c3_mean)
    print(f"  CGG_synergistic = {cgg_syn:+.3f}")
    print(f"\n✅ ALL CELLS DONE in {elapsed/3600:.1f}h — run: python analyze_results.py")
