# CARC reference

Facts about running Fidelis on USC CARC's Discovery cluster. Rules for how Claude works with the cluster live in `CLAUDE.md`; this file is the lookup table behind them.

Sources:
- **[CARC]**: the CARC user guides (https://www.carc.usc.edu/user-guides), read 2026-10-28.
- **[Places]**: notes from the user's earlier Places365 project on CARC (plain sbatch + conda, V100s only). These are observations from one project, not CARC policy.

Anything marked **open** still needs a decision or a check.

## Access

- The user runs every cluster command; Claude has no SSH access (decision, see `CLAUDE.md`).
- Login: `discovery.usc.edu`, or a specific login node (`discovery1.usc.edu` / `discovery2.usc.edu`). Off campus needs the USC VPN. [CARC]
- OnDemand (https://ondemand.carc.usc.edu) gives a JupyterLab terminal on a compute node, and `sbatch` works from it. [Places]
- Login nodes have a per-user limit of 64 processes, 4 CPU cores, and 32 GB memory. Exceeding it can lock you out of the cluster. Heavy work left running on a login node may be killed without warning. [CARC]
- VS Code Remote SSH can trip the process limit and cause a lockout. CARC recommends the SSH-FS extension instead. [CARC]
- Repeated non-interactive SSH attempts from Windows caused "Too many authentication failures". Git-for-Windows SSH can't share connections (no ControlMaster). [Places]
- CARC supports Claude Code on compute nodes (not login nodes, not from `/home`), with a CARC safety harness that prompts before `sbatch` and `git push`. That's an option if we ever move off "the user runs everything". [CARC]

## Account and allocation

- Project IDs look like `<PI_username>_<id>`. `myaccount` lists your accounts, your default account, and your SU balance. [CARC]
- **Always pass `--account` explicitly.** In the Places project the user's *default* account belonged to another group, so a job without `--account` would have charged the wrong group. [Places]
- **Fidelis charges the course account `irahbari_1147`** (decision, 2026-10-28). Use `#SBATCH --account=irahbari_1147` on every job and in every OnDemand form.
- As of 2026-10-28 (`myaccount`): a 60,000,000 SU limit with 59,815,468 remaining. The course shares it, so check `myaccount` for the live balance.
- The course project has two allocations in the CARC User Portal: a compute allocation (SUs, Slurm account `irahbari_1147`) and a storage allocation (`/project2/irahbari_1147`).
- **Both allocations end 2027-07-01** (CARC User Portal). Before then, copy final results and anything expensive to regenerate (splat checkpoints, fine-tuned weights, Monte Carlo outputs) somewhere you own. Raw USGS data doesn't need saving, because the download script rebuilds it.
- **The default account is still another group's account** (confirmed 2026-10-28). A job without `--account` charges that group. Don't use its storage or SUs for Fidelis.

### SU cost per minute [CARC]

| Resource, 1 minute | SUs |
|---|---|
| 1 CPU | 1 |
| 4 GB memory | 1 |
| 1 L40S, A100, or A40 | 8 |
| 1 V100 or P100 | 4 |

SUs are charged for what you *reserve*, not what you use. 1 CPU + 8 GB for an hour is 180 SUs.

## Partitions and limits [CARC]

| Partition | Max run time | Max concurrent GPUs | Max jobs running | Max jobs queued |
|---|---|---|---|---|
| `gpu` | 48 h | 36 | 36 | 100 |
| `main` + `epyc-64` | 48 h | 36 | 100 | 5,000 |
| `oneweek` | 168 h | none | 50 | 50 |
| `largemem` | 168 h | none | 3 | 10 |
| `debug` | 1 h | 4 | 5 | 5 |

The 36-GPU cap is per user, across all jobs.

**Gotcha for Snakemake:** the `gpu` partition allows only 100 *queued* jobs per user. One Slurm job per Monte Carlo sample would hit that cap at once. Samples have to be batched into jobs (spec story 33), and Snakemake's job limit has to stay under the cap.

## GPUs [CARC]

| Type (for requests) | Model | Memory | FP32 | Max per node | Partitions |
|---|---|---|---|---|---|
| `l40s` | L40S | 48 GB | 91.6 TFLOPS | 3 | gpu |
| `a100` | A100 | 40 or 80 GB | 19.5 TFLOPS | 2 | gpu |
| `a40` | A40 | 48 GB | 37.4 TFLOPS | 2 | gpu, debug |
| `v100` | V100 | 32 GB | 14 TFLOPS | 2 | gpu |
| `p100` | P100 | 16 GB | 9.3 TFLOPS | 2 | gpu, debug |

Request syntax, per CARC:

```bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a40:1
```

To pick an A100 size, add `--constraint=a100-40gb` or `--constraint=a100-80gb`. The Places project used `--gres=gpu:v100:1` instead, and that pinned the model reliably. [Places]

`noderes -c -g` shows GPU node resources; `noderes -f -g` shows GPUs free right now. [CARC]

**Why pinning matters for Fidelis:** spec story 8 records each fidelity level's cost together with the GPU type, and MXMCPy uses those costs to split samples across fidelity levels. If jobs drift across GPU models, the costs stop being comparable and the sample split is wrong. Every timed job must pin one type, and for A100s one memory size too.

- V100 has no bf16 support, so mixed precision there means fp16 with a GradScaler. [Places]
- **Open:** which GPU type to standardize on. L40S has by far the highest FP32 rate per SU of anything listed, which suits splat rendering. We won't know until we benchmark real Fidelis renders on it, though.

## Storage [CARC]

| Path | Quota | Backed up | Use in Fidelis |
|---|---|---|---|
| `/home1/$USER` | 100 GB, 1.91 M files | Yes, plus 2 weeks of snapshots | Config files only. CARC says not to run I/O-heavy jobs or coding agents from here. |
| `/project2/irahbari_1147` | 1 TiB, 6 M files, shared by the whole course. 30 GiB used as of 2026-10-28. | Snapshots only (2 weeks) | Raw USGS data, splat checkpoints, final results. Anything expensive to regenerate. Fidelis lives at `/project2/irahbari_1147/ckuizon/fidelis/`. |
| `/scratch1/$USER` | 10 TB, 20 M files | No | Intermediate renders, uv cache, run outputs that the workflow can rebuild |
| `$TMPDIR` | Node-local, RAM-backed | No | Per-job staging. It counts against `--mem`. [Places] |

- **`/scratch1` gets purged whenever the file system is over 80% full.** CARC doesn't say which files go first, so treat everything there as disposable. [CARC]
- `/project2` is all-flash NVMe storage. [CARC]
- **Many small files on `/scratch1` are slow:** about 22 ms per file open in the Places project. The fix was to keep data packed and unpack it into `$TMPDIR` at job start, which raised throughput from about 400 to about 2,190 images/s. [Places]
- **`/scratch1` dropped connections** on a few specific nodes, with `[Errno 108]` and `[Errno 5]` errors. Write atomically (temp file, then rename), retry writes, and exclude bad nodes with `--exclude` (or the `SBATCH_EXCLUDE` environment variable). Find bad nodes by adding `NodeList` to `sacct` output for failed jobs. [Places]
- **Open:** whether `/project2` has the same small-file slowness. Places only measured `/scratch1`.

## Software environment

- Modules: `module spider cuda` / `module spider cudnn` list the CUDA versions available. [CARC] The Places project loaded no CUDA module, because PyTorch wheels bundle their own CUDA runtime. [Places]
- **Directory layout** (decided 2026-10-28). The rule: if the workflow can rebuild a file cheaply, it goes on scratch; otherwise it goes on project2.
  - `/project2/irahbari_1147/ckuizon/fidelis/` holds the git clone, its `.venv`, raw USGS data, splat checkpoints, fine-tuned weights, and final results.
  - `/scratch1/ckuizon/fidelis/` holds the uv cache, intermediate renders, per-batch Monte Carlo outputs, and sweep logs.
- `~/.bashrc` sets `UV_CACHE_DIR=/scratch1/ckuizon/fidelis/uv-cache` and `UV_LINK_MODE=copy`. The cache and the venv sit on different file systems, so uv can't hardlink between them. Copying also means a scratch purge can't break the venv.
- uv 0.13.6 is installed at `~/.local/bin` (2026-10-28). If a shell can't find `uv`, run `source $HOME/.local/bin/env`. **Open:** `uv sync` hasn't been tested on CARC yet (setup steps below). The lock pins torch `cu134`, which needs an NVIDIA driver from the R580 series or newer (CUDA 13 minor-version compatibility, per NVIDIA's CUDA compatibility docs); check `nvidia-smi` on a GPU node. See `docs/environment.md`.
- The repo is cloned at `/project2/irahbari_1147/ckuizon/fidelis`. The workflow: edit locally, push, then `git pull` on CARC. If you ever commit on CARC, run `git config core.hooksPath .githooks` in the clone first.
- To install packages or debug on the cluster, use an interactive debug node (`salloc -p debug -c 4`), not the login node. [CARC]
- `os.cpu_count()` reports every CPU on the node, not the job's share. Use `len(os.sched_getaffinity(0))` instead. [Places]

## uv environment on CARC

One-time setup, from the clone on `/project2`. The cache goes on `/scratch1` and the venv sits in the clone, per the directory layout above. Run `uv sync` on a debug node, not the login node: it unpacks multi-GB CUDA wheels, which can exceed the login node's limits.

```bash
grep -E 'UV_CACHE_DIR|UV_LINK_MODE' ~/.bashrc   # expect /scratch1/ckuizon/fidelis/uv-cache and copy
salloc -p debug -c 4 --mem 16G --account=irahbari_1147 -t 1:00:00
cd /project2/irahbari_1147/ckuizon/fidelis
git pull
uv sync                                          # dev + workflow groups: pytest, Snakemake, SLURM plugin
uv run pytest                                    # GPU-free suite
exit                                             # release the debug node
```

Re-run `uv sync` after any `git pull` that changes `uv.lock`. A `/scratch1` purge only costs a re-download, because `UV_LINK_MODE=copy` means the venv doesn't point into the cache.

## Workflow profiles

`workflow/Snakefile` declares what each rule needs (GPUs, memory, time). A profile decides where it runs:

- `profiles/local`: the laptop GPU, one GPU job at a time.
- `profiles/carc`: the SLURM executor plugin. It sets `slurm_account=irahbari_1147` and the `gpu` partition for every job, pins each GPU rule's model (`smoke` → `l40s`), loads environment modules, keeps logs of successful jobs, and caps Snakemake at 20 jobs, well under the 100-job queue cap.

Two plugin behaviors drive those choices (snakemake-executor-plugin-slurm 2.8.0, read from its source 2026-10-31):

- **With no `slurm_account`, the plugin guesses one from your recent jobs** (`get_account_arg` in `__init__.py`). On CARC that could be the other group's account, so the profile always sets it. `tests/test_workflow.py` checks every planned job.
- **Setting `gpu_model` makes the plugin request a GPU** (`--gpus=<model>:1`, `set_gres_string` in `utils.py`) even for a rule that declares none. So the model is pinned per rule in `set-resources`, never in `default-resources`.

On CARC, launch with `uv run --no-sync snakemake ...`. uv puts the venv on `PATH`, and the plugin submits with `--export=ALL` (`submit_string.py`), so jobs should inherit it and run `fidelis-smoke` and the job-side Snakemake from the venv (unverified on CARC until the first run). `--no-sync` stops uv from re-syncing the venv on the login node when `uv.lock` has changed, which would unpack multi-GB CUDA wheels there. Sync on a debug node instead (above).

## Cluster smoke test

The `smoke` rule runs `nvidia-smi` (driver version and GPU model) and then `fidelis-smoke` (the five library checks). It has no output file, so it runs every time it's requested. On CARC the PASS/FAIL lines land in the job log at `.snakemake/slurm_logs/rule_smoke/<jobid>.log`.

Before `smoke` is submitted, the `preflight` rule runs `fidelis-preflight` on the login node (a `localrule`), with the same modules as `smoke`. It checks that `nvcc` is on `PATH`, that it shares torch's CUDA major version, and that it compiles a trivial kernel with the loaded host compiler. It takes seconds and needs no GPU. If it fails, nothing is submitted, and its PASS/FAIL lines print in the terminal. Both profiles set `forcerun: [preflight]`, so it runs on every invocation, not only the first. It can't check anything that needs a GPU: the driver, or gsplat's build for the node's GPU architecture. The first L40S run (job 13632519) would have been caught by it.

Cost: the rule reserves 1 GPU, 4 CPUs, and 16 GB for at most 30 minutes, which is at most 480 SUs per run (8 + 4 + 4 SUs per minute).

From the clone on a login node:

```bash
cd /project2/irahbari_1147/ckuizon/fidelis
uv run --no-sync snakemake --profile profiles/carc smoke
```

Then the A100, after the L40S run finishes (see the gsplat note below):

```bash
uv run --no-sync snakemake --profile profiles/carc smoke --set-resources smoke:gpu_model=a100
```

This doesn't pin the A100 memory size (40 or 80 GB). The smoke test isn't timed, so the size doesn't matter here. Timed rules must also set the plugin's `constraint` resource (`a100-40gb` or `a100-80gb`, passed as `-C`).

`--set-resources` on the command line replaces the profile's whole `set-resources` list; it doesn't merge with it. A dry run on 2026-10-31 dropped a profile entry `smoke:runtime=45` when given `--set-resources smoke:gpu_model=a100`. That's why the profile keeps only GPU models there. Once there are more GPU rules, an override has to repeat every pin it wants to keep.

**Run one GPU type at a time.** gsplat compiles into one shared cache (`~/.cache/torch_extensions`), built for the GPU it compiled on. Two jobs compiling at once would race on the same directory: gsplat deletes its build directory's lock file on every load (`gsplat/cuda/build.py`, `build_and_load_gsplat`), so the lock doesn't protect concurrent builds. **Unverified:** that a run on another GPU type recompiles cleanly. torch's extension version hash ignores the GPU architecture, so it relies on ninja noticing the changed `-gencode` flags. A compile took about 6 minutes on the laptop (`docs/environment.md`). If the A100 run fails with "no kernel image is available", it loaded the L40S build. Delete `~/.cache/torch_extensions/py312_cu134/gsplat_cuda` and rerun.

### GPU types that passed

| GPU type | Date | Driver | Result | Job log |
|---|---|---|---|---|
| `l40s` | | | **open**: not run yet | |
| `a100` | | | **open**: not run yet | |

### CARC-specific workarounds

- **CUDA toolkit module and its prerequisites.** gsplat needs `nvcc` from a CUDA toolkit with the same major version as torch's CUDA (13); without it gsplat prints "No CUDA toolkit found" and its check fails. CARC's newest is `cuda/13.0.3` (`module spider cuda`, 2026-10-31; the other versions go from 10.0 to 12.9). It only loads after `ver/2607` and `gcc/15.3.0` (`module spider cuda/13.0.3`, 2026-10-31), so the rule loads all three, in that order, through Snakemake's `envmodules`. torch's `cu134` build with a 13.0 toolkit passes torch's major-version check. Whether nvcc 13.0 accepts GCC 15 as a host compiler and compiles gsplat cleanly is unverified until a cluster run passes.
- **A failed module load doesn't stop the job.** Snakemake 9.27.1 runs `module purge && module load ...; <command>` (`snakemake/deployment/env_modules.py`). The `;` means the command runs even when the load fails, so the only trace is an Lmod error near the top of the job log. The first L40S run (job 13632519, 2026-10-31) loaded `gcc/13.3.0` and `cuda/13.0.3` together, Lmod refused `cuda/13.0.3` ("exist but cannot be loaded as requested"), and gsplat failed while the other four checks passed. The `preflight` rule now checks this on the login node before every submission (see "Cluster smoke test").
- **Driver.** The `cu134` torch build needs an R580-or-newer driver. The L40S node c04-02 runs 580.159.04 (`nvidia-smi` in job 13632519's log, 2026-10-31).

## Running the Snakemake controller

The Snakemake controller process has to stay alive for the whole workflow. It's light, but it runs for a long time. Options:
- **tmux on a fixed login node** (`discovery1.usc.edu`). CARC documents tmux on login nodes (`module load gcc/13.3.0 tmux/3.4`), but login nodes reboot during maintenance. That's survivable, because Snakemake reruns only missing steps. [CARC]
- **Inside a long CPU job** on `main` (48 h) or `oneweek` (168 h). It's never killed by a login-node cleanup, but it costs SUs while it waits.
- **Open:** pick one. Also check that the SLURM plugin's status polling stays well under the login node's 64-process limit.

## Monitoring commands

| Purpose | Command |
|---|---|
| Queue | `squeue --me` or `myqueue` |
| Job history | `sacct -u $USER -S now-2days --format=JobID,JobName%15,State,Elapsed,ExitCode,NodeList,MaxRSS -X` |
| Efficiency of a finished job | `seff <jobid>` |
| CARC summaries | `jobhist`, `jobinfo <jobid>` |
| Account and SUs | `myaccount` |
| Quotas | `myquota` |

Sources: [CARC] and [Places].
