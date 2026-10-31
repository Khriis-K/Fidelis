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
- uv 0.13.6 is installed at `~/.local/bin` (2026-10-28). If a shell can't find `uv`, run `source $HOME/.local/bin/env`. **Open:** `uv sync` hasn't been tested on CARC yet. The lock pins torch `cu134`, which needs an NVIDIA driver from the R580 series or newer (CUDA 13 minor-version compatibility, per NVIDIA's CUDA compatibility docs); check `nvidia-smi` on a GPU node. See `docs/environment.md`.
- The repo is cloned at `/project2/irahbari_1147/ckuizon/fidelis`. The workflow: edit locally, push, then `git pull` on CARC. If you ever commit on CARC, run `git config core.hooksPath .githooks` in the clone first.
- To install packages or debug on the cluster, use an interactive debug node (`salloc -p debug -c 4`), not the login node. [CARC]
- `os.cpu_count()` reports every CPU on the node, not the job's share. Use `len(os.sched_getaffinity(0))` instead. [Places]

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
