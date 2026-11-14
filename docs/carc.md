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
- **Pinned: A40** for every cost measurement from issue #10 onward (the user's call, 2026-11-11). Request it with `--gres=gpu:a40:1` and set `TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_a40`. Why A40:
  - It's the only type that has had free cards when we looked.
  - The descent profile already ran on it, so all the cost numbers so far come from one GPU type.
  - L40S has the highest FP32 rate per SU of anything listed, but our L40S job never got a node.
  
  The A40 is on both `gpu` and `debug`; debug jobs are limited to 1 h. Earlier timings on L40S (the plain MC and splat runs) aren't comparable with A40 costs.
- **L40S availability is an input to that choice.** The evening of 2026-11-10, `squeue -p gpu` listed 78 pending L40S jobs (65 on Priority, 13 on dependencies) and 7 running, using 9 L40S GPUs on nodes b17-16, b18-16, b18-17 and b18-18. A 30-minute single-L40S job of ours (14002051) sat pending on Priority behind them. One snapshot, not a wait-time measurement; the user reports L40S waits have been long lately. Job 14002051 ended `CANCELLED` without ever getting a node (`sacct`, checked 2026-11-11), so there's no L40S descent profile.
- **Snapshot, 2026-11-11.** `noderes -f -g` showed 2 free A40s (b11-09, debug partition, idle) and 2 free P100s, and nothing free on the `gpu` partition. On `squeue -p gpu`, single-GPU jobs that named a type had these pending and running counts: A100 158 / 8, A40 114 / 26, L40S 30 / 3. Another 577 untyped single-GPU jobs were pending. One snapshot again.
- **Endeavour: not an option (checked 2026-11-11).** The user can log in to `endeavour2`, CARC's condo cluster. Every GPU partition in `noderes -f -g` there is named after an owning group (`ruska`, `cryoem`, `fibril`, `qcbgpu`, ...), apart from one V100 node on `shared`; there is no `gpu` partition (`squeue -p gpu` printed nothing). It keeps its own job database: `sacct -j 14002051` (a Discovery job) found nothing. The CARC User Portal lists no Endeavour allocation for the user, only Discovery ones (`irahbari_1147`, `ruishanl_1185`). `sacctmgr show assoc user=$USER`, run on endeavour2, lists only two associations, both on cluster `discovery` (`irahbari_1147` and `ruishanl_1185`) and none on Endeavour, so neither account can submit jobs there. Revisit only if the course or lab gets an Endeavour condo allocation.

## Storage [CARC]

| Path | Quota | Backed up | Use in Fidelis |
|---|---|---|---|
| `/home1/$USER` | 100 GB, 1.91 M files | Yes, plus 2 weeks of snapshots | Config files only. CARC says not to run I/O-heavy jobs or coding agents from here. |
| `/project2/irahbari_1147` | 1 TiB, 6 M files, shared by the whole course. 30 GiB used as of 2026-10-28; 82 GiB as of 2026-11-11 (user's quota page). | Snapshots only (2 weeks) | Raw USGS data, splat checkpoints, final results. Anything expensive to regenerate. Fidelis lives at `/project2/irahbari_1147/ckuizon/fidelis/`. |
| `/scratch1/$USER` | 10 TB, 20 M files | No | Intermediate renders, uv cache, run outputs that the workflow can rebuild |
| `$TMPDIR` | Node-local, RAM-backed | No | Per-job staging. It counts against `--mem`. [Places] |

- **`/scratch1` gets purged whenever the file system is over 80% full.** CARC doesn't say which files go first, so treat everything there as disposable. [CARC]
- `/project2` is all-flash NVMe storage. [CARC]
- **Many small files on `/scratch1` are slow:** about 22 ms per file open in the Places project. The fix was to keep data packed and unpack it into `$TMPDIR` at job start, which raised throughput from about 400 to about 2,190 images/s. [Places]
- **`/scratch1` dropped connections** on a few specific nodes, with `[Errno 108]` and `[Errno 5]` errors. Write atomically (temp file, then rename), retry writes, and exclude bad nodes with `--exclude` (or the `SBATCH_EXCLUDE` environment variable). Find bad nodes by adding `NodeList` to `sacct` output for failed jobs. [Places]
- **Open:** whether `/project2` has the same small-file slowness. Places only measured `/scratch1`.
- **Compute nodes have no internet.** CARC says to "complete these transfers on the login or transfer nodes separately from Slurm jobs" ([Transferring data using the command line](https://www.carc.usc.edu/user-guides/research-data-management/transferring-data/transferring-data-command-line), last updated 2024-03-11). The page names `hpc-transfer1.usc.edu` as a transfer node and suggests running long transfers in tmux. It doesn't state limits for transfer nodes. [CARC]
  - So `download_mosaic` is a localrule. It runs wherever Snakemake was launched, which is the login node in tmux (see "Running the Snakemake controller"). It's one streaming core, well under the login node's 4-core limit. The tile and map steps don't need internet, so they still go to `main`.
  - **Worked (2026-11-01):** `snakemake --profile profiles/carc jezero`, from tmux on `discovery1`, finished all 6 steps in 4 min 40 s. That covers both downloads on the login node, plus `jezero_tile` and `jezero_matching_map` as `main` jobs. Both mosaics matched the SHA-256 hashes pinned on the laptop, and the tile is byte-for-byte the same size (`dtm.tif` 16,076,792 B, `ortho.tif` 64,305,031 B). In the log, the localrule `jezero` shows `slurm_partition=gpu` from `default-resources`. That's harmless, because localrules are never submitted.
  - Alternative: download on the laptop and `rsync -rltvh data/jezero/raw/ ckuizon@hpc-transfer1.usc.edu:/project2/irahbari_1147/ckuizon/fidelis/data/jezero/raw/`. The pinned checksums verify the files on arrival, so the next `snakemake jezero` re-hashes them and skips the download.

## Software environment

- Modules: `module spider cuda` / `module spider cudnn` list the CUDA versions available. [CARC] The Places project loaded no CUDA module, because PyTorch wheels bundle their own CUDA runtime. [Places]
- **Directory layout** (decided 2026-10-28). The rule: if the workflow can rebuild a file cheaply, it goes on scratch; otherwise it goes on project2.
  - `/project2/irahbari_1147/ckuizon/fidelis/` holds the git clone, its `.venv`, raw USGS data, splat checkpoints, fine-tuned weights, and final results.
  - `/scratch1/ckuizon/fidelis/` holds the uv cache, intermediate renders, per-batch Monte Carlo outputs, and sweep logs.
- `~/.bashrc` sets `UV_CACHE_DIR=/scratch1/ckuizon/fidelis/uv-cache` and `UV_LINK_MODE=copy`. The cache and the venv sit on different file systems, so uv can't hardlink between them. Copying also means a scratch purge can't break the venv.
- uv 0.13.6 is installed at `~/.local/bin` (2026-10-28). If a shell can't find `uv`, run `source $HOME/.local/bin/env`. `uv sync` works on CARC (setup steps below; the L40S smoke test passed from that venv on 2026-11-01). The lock pins torch `cu134`, which needs an NVIDIA driver from the R580 series or newer (CUDA 13 minor-version compatibility, per NVIDIA's CUDA compatibility docs); see "Driver" below. See `docs/environment.md`.
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

The venv uses a uv-managed Python 3.12 (`python-preference = "only-managed"` in `pyproject.toml`), not the node's `/usr/bin/python3.12`. The first venv was built on the system Python, which gave two problems: its interpreter differed between nodes (3.12.15 on e23-02, 3.12.14 on d11-03, so uv warned about an incompatible environment), and the compute nodes lack its headers, so gsplat's build failed with `Python.h: No such file or directory` (job 13639170). A venv made before that change must be deleted once (`rm -rf .venv`) before `uv sync`; uv installs the interpreter under `~/.local/share/uv/python`.

Re-run `uv sync` after any `git pull` that changes `uv.lock` or `pyproject.toml`. A new command in `[project.scripts]` changes only `pyproject.toml`, and it isn't installed until a sync: `fidelis-scene` (#33) failed with "command not found" in job 13673724 (2026-11-02) for that reason. `uv sync --dry-run` lists what a sync would change; if it's only the `fidelis` package itself, there's nothing to unpack and it's light enough for the login node. A `/scratch1` purge only costs a re-download, because `UV_LINK_MODE=copy` means the venv doesn't point into the cache.

## Workflow profiles

`workflow/Snakefile` declares what each rule needs (GPUs, memory, time). A profile decides where it runs:

- `profiles/local`: the laptop GPU, one GPU job at a time.
- `profiles/carc`: the SLURM executor plugin. It sets `slurm_account=irahbari_1147` and the `gpu` partition for every job, pins each GPU rule's model (`smoke` → `l40s`), loads environment modules, keeps logs of successful jobs, and caps Snakemake at 20 jobs, well under the 100-job queue cap.

Two plugin behaviors drive those choices (snakemake-executor-plugin-slurm 2.8.0, read from its source 2026-10-31):

- **With no `slurm_account`, the plugin guesses one from your recent jobs** (`get_account_arg` in `__init__.py`). On CARC that could be the other group's account, so the profile always sets it. `tests/test_workflow.py` checks every planned job.
- **Setting `gpu_model` makes the plugin request a GPU** (`--gpus=<model>:1`, `set_gres_string` in `utils.py`) even for a rule that declares none. So the model is pinned per rule in `set-resources`, never in `default-resources`.

On CARC, launch with `uv run --no-sync snakemake ...`. uv puts the venv on `PATH`, and the plugin submits with `--export=ALL` (`submit_string.py`), so jobs inherit it and run `fidelis-smoke` and the job-side Snakemake from the venv (job 13649053's log shows the job-side Snakemake running from `.venv/lib/python3.12/site-packages`). `--no-sync` stops uv from re-syncing the venv on the login node when `uv.lock` has changed, which would unpack multi-GB CUDA wheels there. Sync on a debug node instead (above).

## Cluster smoke test

The `smoke` rule runs `nvidia-smi` (driver version and GPU model) and then `fidelis-smoke` (the five library checks). It has no output file, so it runs every time it's requested. On CARC the PASS/FAIL lines land in the job log at `.snakemake/slurm_logs/rule_smoke/<jobid>.log`.

Before `smoke` is submitted, the `preflight` rule runs `fidelis-preflight` on the login node (a `localrule`), with the same modules as `smoke`. It checks that `nvcc` is on `PATH`, that it shares torch's CUDA major version, that it compiles a trivial kernel with the loaded host compiler, and that the venv's Python has the `Python.h` torch compiles extensions against. It takes seconds and needs no GPU. If it fails, nothing is submitted, and its PASS/FAIL lines print in the terminal. Both profiles set `forcerun: [preflight]`, so it runs on every invocation, not only the first. It can't check anything that needs a GPU: the driver, or gsplat's build for the node's GPU architecture. The first L40S run (job 13632519) would have been caught by it.

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

**Never pass `--default-resources` on the command line.** It replaces the profile's list the same way, and that list holds `slurm_account` and `slurm_partition`. A dry run with `--default-resources mem_mb=1000` (2026-11-01) planned `preflight` with neither, so the plugin would guess the account (see above). `tests/test_workflow.py` checks the profile, not the command line, so it can't catch this.

**Run one GPU type at a time.** gsplat compiles into one shared cache (`~/.cache/torch_extensions`), built for the GPU it compiled on. Two jobs compiling at once would race on the same directory: gsplat deletes its build directory's lock file on every load (`gsplat/cuda/build.py`, `build_and_load_gsplat`), so the lock doesn't protect concurrent builds. Switching GPU type recompiles cleanly: torch's extension version hash ignores the GPU architecture, but ninja notices the changed `-gencode` flags. The A100 run right after the L40S run (job 13652238) rebuilt gsplat for sm_80 and passed, taking 4 min 23 s against 38 s for the L40S run that reused its build. The cache now holds the A100 build, so the next L40S run rebuilds too. If a run fails with "no kernel image is available", it loaded the L40S build. Delete `~/.cache/torch_extensions/py312_cu134/gsplat_cuda` and rerun.

### GPU types that passed

| GPU type | Date | Driver | Result | Job log |
|---|---|---|---|---|
| `l40s` | 2026-11-01 | 580.159.04 | 5/5 pass (c04-02, 38 s) | job 13649053, `.snakemake/slurm_logs/rule_smoke/13649053.log` |
| `a100` | 2026-11-01 | 580.159.04 | 5/5 pass (b02-20, A100-PCIE-40GB, 4 min 23 s including the gsplat rebuild) | job 13652238, `.snakemake/slurm_logs/rule_smoke/13652238.log` |

`a40`: not smoke-tested, but the per-stage descent profile ran on one (debug partition, 2026-11-10). gsplat built for sm_86 in 145 s and rendered; 30 descents all located (docs/benchmarks/descent-profile.md). It built into its own cache by setting `TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_a40`: torch's `_get_build_directory` reads that variable and gsplat builds through it (`gsplat/cuda/build.py`). A separate cache per GPU type avoids the race and the rebuild back and forth described above. Not yet checked: that the shared L40S cache was left untouched.

`v100` and `p100` can't run this stack: CUDA 13 dropped Pascal and Volta (sm_60–sm_70), so torch `cu134` has no kernels for them (NVIDIA CUDA 13.0 release notes; not tried on CARC).

### CARC-specific workarounds

- **CUDA toolkit module and its prerequisites.** gsplat needs `nvcc` from a CUDA toolkit with the same major version as torch's CUDA (13); without it gsplat prints "No CUDA toolkit found" and its check fails. CARC's newest is `cuda/13.0.3` (`module spider cuda`, 2026-10-31; the other versions go from 10.0 to 12.9). It only loads after `ver/2607` and `gcc/15.3.0` (`module spider cuda/13.0.3`, 2026-10-31), so the rule loads all three, in that order, through Snakemake's `envmodules`. torch's `cu134` build with a 13.0 toolkit passes torch's major-version check, and nvcc 13.0 with GCC 15 builds gsplat (see below).
- **A failed module load doesn't stop the job.** Snakemake 9.27.1 runs `module purge && module load ...; <command>` (`snakemake/deployment/env_modules.py`). The `;` means the command runs even when the load fails, so the only trace is an Lmod error near the top of the job log. The first L40S run (job 13632519, 2026-10-31) loaded `gcc/13.3.0` and `cuda/13.0.3` together, Lmod refused `cuda/13.0.3` ("exist but cannot be loaded as requested"), and gsplat failed while the other four checks passed. The `preflight` rule now checks this on the login node before every submission (see "Cluster smoke test").
- **gsplat's host C++ on Linux.** Job 13639170 (L40S, b18-17, 2026-11-01) loaded the modules and compiled all of gsplat's CUDA files, then failed on two host `.cpp` files: `'::isfinite' has not been declared` (a bug in our fork, fixed in `0270f88`; see docs/environment.md) and the missing `Python.h` above. The preflight now checks for `Python.h`; it can't catch C++ errors in gsplat itself short of building it. With both fixes, gsplat's full build (all 51 steps, including the link) succeeded on a CPU-only debug node in 300 s (2026-11-01), and the L40S job 13649053 then rendered with it (gsplat check passed).
- **Build gsplat on a CPU node before a GPU job.** gsplat's JIT build needs no GPU if `TORCH_CUDA_ARCH_LIST` names the target chip, so test a build on a debug allocation (which gets through the queue in minutes) instead of a GPU slot:
  ```bash
  salloc -p debug -c 8 --mem 32G --account=irahbari_1147 -t 1:00:00
  cd /project2/irahbari_1147/ckuizon/fidelis && module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && TORCH_CUDA_ARCH_LIST="8.9+PTX" uv run --no-sync python -c "from fidelis.gsplat_loader import import_gsplat; import_gsplat(); from gsplat.cuda._backend import _C; print('GSPLAT BUILT:', _C)"
  ```
  Go through `import_gsplat()` so the build flags match the GPU job's; otherwise gsplat wipes the cache when the job loads it. `8.9+PTX` is the L40S (sm_89); torch's own auto-detection on an L40S produces the same `-gencode` flags, so the GPU job reuses this build: job 13649053 printed no build output and finished all five checks in 38 s, against 300 s for the build alone. Don't run it while a GPU job may be building gsplat: the two would race on the same cache directory.
- **Driver.** The `cu134` torch build needs an R580-or-newer driver. The L40S node c04-02 runs 580.159.04 (`nvidia-smi` in job 13632519's log, 2026-10-31).
- **Harmless warning: `jitc_llvm_init(): LLVM API initialization failed`.** Mitsuba's JIT (Dr.Jit) tries to load LLVM for its CPU backend at import and finds none on the compute node. Fidelis uses the `cuda_ad_rgb` variant, which doesn't need LLVM, and the Mitsuba check passes with the warning printed (job 13649053). It would matter only for an `llvm_*` variant.

## Plain Monte Carlo on CARC

The wide scene (#33) doesn't fit the laptop: its matching map gives 70,726 keypoints on the laptop's GPU (14,290 on the 10 km tile; measured 2026-11-02; the A40 counts 70,749 on the same map, `docs/benchmarks/ctx-map-scan.md`), and LightGlue ran out of the RTX 5070's 8 GB matching a descent image against them. The `plain_mc` rule is pinned to `l40s` (48 GB) in `profiles/carc`.

- **Weights first, on the login node.** ALIKED and LightGlue download their weights from GitHub on first use (`torch.hub.load_state_dict_from_url` in `lightglue/aliked.py` and `lightglue/lightglue.py`), and compute nodes have no internet. Building both models once on the login node caches `aliked-n16.pth` and `aliked_lightglue_v0-1_arxiv.pth` in `~/.cache/torch/hub/checkpoints`, where the job finds them. CPU only, a few seconds.
- **One BLAS thread on the login node.** Without `OPENBLAS_NUM_THREADS=1`, OpenBLAS starts a thread per CPU it sees (32 on `discovery1`). Threads count against the login node's 64-process limit, so `pthread_create` failed from thread 22 on and the `cv2` import hung (2026-11-02). The `RLIMIT_NPROC 1546766` in its message is the ulimit, not the cap that was hit. `OMP_NUM_THREADS=1` does the same for torch's OpenMP pool.
- **The wide scene is built by the same command.** `plain_mc` depends on `jezero`, so a clone that has only the 10 km tile also downloads the two CTX products (46 MB, on the login node) and runs `jezero_scene`, `jezero_mesh` and `jezero_matching_map` on `main`.
- **Cost:** the rule reserves 1 GPU, 4 CPUs and 16 GB for at most 60 minutes, so at most 960 SUs (8 + 4 + 4 SUs per minute).
- **Cost per sample is L40S time,** not comparable with the laptop runs in `docs/benchmarks/plain-mc.md`.
- **Worked (2026-11-02):** job 13674053 on b18-18 (NVIDIA L40S) ran all 20 samples in 40 s (`sacct` Elapsed for the job step), with host memory peaking at 2.95 GB of the 16 GB requested. The scene steps ran first as `main` jobs. Results are in `docs/benchmarks/plain-mc.md` ("Wide-scene run").

From tmux on `discovery1`, in the clone:

```bash
cd /project2/irahbari_1147/ckuizon/fidelis
git pull
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -c "from lightglue import ALIKED, LightGlue; ALIKED(); LightGlue(features='aliked')"
ls ~/.cache/torch/hub/checkpoints
uv run --no-sync snakemake --profile profiles/carc plain_mc -n
uv run --no-sync snakemake --profile profiles/carc plain_mc
```

If `git pull` changed `uv.lock` or `pyproject.toml`, run `uv sync` first ("uv environment on CARC"). The dry run (`-n`) should list `plain_mc` plus whatever scene steps are missing, every job with `slurm_account=irahbari_1147`. The estimate prints at the end of `.snakemake/slurm_logs/rule_plain_mc/<jobid>.log`; the results are `results/plain_mc/records.jsonl` and `estimate.json`.

## Splat scene on CARC (#9)

`snakemake splat` renders the training views (`splat_views`) and trains one splat scene under the config's `splat.sun_index` (`splat_train`); `snakemake plain_mc_splat` then runs plain Monte Carlo on it. `plain_mc_mesh_pinned` runs the same seeds on the mesh under the same pinned sun: `plain_mc` mixes all five suns, so only this run compares the two fidelities on their own. All four are pinned to `l40s` in `profiles/carc`.

- **Outputs go to `data/jezero/splat/sun0/`**, inside the clone on `/project2`: the checkpoint `splat.pt`, the browser export `splat.ply`, `quality.json` (held-out PSNR and SSIM against the mesh, with the constant-image baseline), and `views/` (about 0.6 GB of float16 renders, cheap to regenerate). All git-ignored.
- **gsplat must already be built for the L40S** (see "Build gsplat on a CPU node before a GPU job"). The cache holds whichever GPU type built it last; an A100 build makes the first L40S job rebuild it.
- **New commands need a sync.** This adds `fidelis-splat-views` and `fidelis-splat-train` to `[project.scripts]`, so run `uv sync` (or check with `uv sync --dry-run`) after pulling, or the job fails with "command not found". It did: job 13681953 (2026-11-02) failed in 5 s this way.
- **Cost (reserved, upper bounds):** `splat_views` 1 GPU, 4 CPUs, 16 GB for 60 min: 960 SUs. `splat_train` 1 GPU, 8 CPUs, 64 GB for 45 min: 1,440 SUs. `plain_mc_splat` and `plain_mc_mesh_pinned` as `plain_mc`, 960 SUs each. The first training run took 12 min 49 s (610 s of it training), starting from 7,984,151 Gaussians (one per mesh vertex) and ending with 7,715,069; the limit was 240 min until then. Its host memory peak isn't known (`sacct` showed no `MaxRSS`), so the 64 GB stays.
- **Worked (2026-11-02):** `splat_views` job 13682196 (30 s), `splat_train` job 13682227 (12 min 49 s, gate passed), `plain_mc_splat` (its job ran after the earlier `sacct` listing, so its ID wasn't recorded), `plain_mc_mesh_pinned` job 13681952 (38 s, from the first attempt, which it finished before `splat_views` failed); all on L40S node b17-16. Results in `docs/benchmarks/splat.md`.
- **The controller takes a minute or two to notice a finished job.** The SLURM plugin checks job status periodically, and the profile's `latency-wait: 60` waits for outputs to appear on shared storage. After `splat_train` finished, the controller's log stayed at "has been submitted" for a few minutes before it moved on. Killing it then (it looked stuck) lost only its record of the next job (`plain_mc_splat`), whose results were complete: the next dry run reported "missing metadata" for that rule. Wait at least 5 minutes after a job ends before deciding a controller is stuck.

```bash
cd /project2/irahbari_1147/ckuizon/fidelis
git pull
uv sync --dry-run
uv run --no-sync snakemake --profile profiles/carc splat plain_mc_splat plain_mc_mesh_pinned -n
uv run --no-sync snakemake --profile profiles/carc splat plain_mc_splat plain_mc_mesh_pinned
```

The training log is `.snakemake/slurm_logs/rule_splat_train/<jobid>.log`; its last line gives the held-out worst PSNR and SSIM, the constant image's, and whether the gate passed. The training job succeeds even when the gate fails, so its checkpoint survives to be inspected (Snakemake deletes a failed job's outputs). Then the seam-1 tests on the trained splat, on an L40S:

```bash
srun --account=irahbari_1147 --partition=gpu --gres=gpu:l40s:1 -c 4 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync pytest -m gpu tests/test_simulation_splat.py -v"
```

## CTX map scan on CARC (#36)

The navigator now matches against the real LVS map, which the workflow builds as `data/jezero/lvs_map/map.tif` (`jezero_lvs_map`, CPU only, on `main`). The scan and its control are pre-registered in `docs/benchmarks/ctx-map-scan.md`; run them only as written there. The workflow's own Monte Carlo rules (`plain_mc`, `plain_mc_splat`, `plain_mc_mesh_pinned`, `match_viz`) stay on the old map, so they still reproduce the earlier results and can't spend the old seeds on the new map.

- **New command, so sync first.** This adds `fidelis-lvs-map` to `[project.scripts]`. Without `uv sync` the map job fails with "command not found", as `splat_views` once did.
- **The map first.** The dry run should plan `jezero_lvs_map` and nothing else. If it plans scene or mesh steps too, stop: those outputs have no recorded params ("Config edits and reruns"), and a rebuild would change the scene under the scan.
- **Cost (reserved, upper bounds):** the map job is 1 CPU and 4 GB on `main` for about a minute, a few SUs. Each scan job reserves 1 A40, 8 CPUs and 32 GB, 24 SUs per minute: at most 720 SUs at the 30-minute limit. The failure scan's descents took a median 0.479 s on an A40, so about 5 minutes and 100–150 SUs each is the estimate.

```bash
cd /project2/irahbari_1147/ckuizon/fidelis
git pull
uv sync --dry-run
uv sync
uv run --no-sync snakemake --profile profiles/carc data/jezero/lvs_map/map.tif -n
uv run --no-sync snakemake --profile profiles/carc data/jezero/lvs_map/map.tif
cat data/jezero/lvs_map/map.json
```

`map.json` should give a shape of `[4340, 4350]` and `"nodata_pixels": 0`, as on the laptop. Then the scan, and the control on the old map (same seeds):

```bash
srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/ctx_map_scan --fidelity mesh --first-seed 20000 --samples 300"
srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/ctx_map_scan_control --fidelity mesh --first-seed 20000 --samples 300 --matching-map scene"
```

Each prints its estimate on the last line and writes `records.jsonl` and `estimate.json` into its results folder.

- **Worked (2026-11-12).** The first map job (14062236) failed in seconds with `fidelis-lvs-map: command not found`: the branch was checked out but `uv sync` hadn't run. After the sync the map built with the laptop's shape and bounds. No A40 or L40S was free (`noderes -f -g`), so the scan and control went in as one batch job, which waits in the queue without holding a terminal: `sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 30 -J ctx_map_scan -o results/ctx_map_scan_%j.log --wrap "<the two commands above, without srun, joined by &&>"`. Job 14062475 ran both on b11-09 (A40) in 5 min 23 s, about 129 SUs [Computed]. Results in `docs/benchmarks/ctx-map-scan.md`.
- **The `tac450-th` reservation can't run this stack.** It is 12 nodes (`d14-[06-13,15-18]`), Thursdays 15:00–18:00, open to `irahbari_1147` (`scontrol show reservation tac450-th`, 2026-11-12), but its GPUs are V100s (`Gres=gpu:v100:2`), which CUDA 13 doesn't support. Its 384 CPU cores are usable for CPU-only work.

## Config edits and reruns

Every rule that reads `config/jezero.yaml` takes it as `ancient()`, so the file's timestamp never reruns anything, and lists the values it actually reads as `params`. Snakemake reruns a rule when its params change. So a comment edit, or a change to another rule's setting, reruns nothing; changing `tile.size_m` reruns the scene and everything after it; changing `splat.steps` reruns training. Before this, any config edit rebuilt the whole scene: that's why the first `splat` run (2026-11-02) also reran `jezero_tile`, `jezero_scene`, `jezero_mesh` and `jezero_matching_map`.

**One-time migration.** Outputs built before this change have no params on record, so Snakemake reads that as "changed" and plans a rebuild, the splat training included. Clear their old records once, after the current run has finished (never while a workflow is running), in the clone:

```bash
uv run --no-sync snakemake --profile profiles/carc --list-params-changes splat plain_mc_splat plain_mc_mesh_pinned plain_mc
uv run --no-sync snakemake --profile profiles/carc --cleanup-metadata $(uv run --no-sync snakemake --profile profiles/carc --list-params-changes splat plain_mc_splat plain_mc_mesh_pinned plain_mc 2>/dev/null | grep -v '^Using\|^Building')
uv run --no-sync snakemake --profile profiles/carc splat plain_mc_splat plain_mc_mesh_pinned -n
```

The first line lists what would be cleared. The last should plan only `preflight` (and `plain_mc`, if its results are older than the scene rebuilt on 2026-11-02). It only deletes Snakemake's bookkeeping in `.snakemake/metadata`, never data. Checked on the laptop's copy of the workflow (2026-11-02): before it, a dry run planned the whole scene again; after it, only a `plain_mc` whose results really did predate the scene.

**After the migration, a config change can't rerun those outputs on its own.** With their records cleared, Snakemake has nothing to compare new params with: the dry run on CARC (2026-11-02) planned only `preflight` and reported "12 jobs have missing provenance/metadata so that it in part cannot be used to trigger re-runs" (`download_mosaic`, the four scene rules, `splat_views`, `splat_train`, `plain_mc_splat`, `plain_mc_mesh_pinned`). Until a rule's outputs are rebuilt, and so recorded afresh, changing a value it reads needs `--forcerun <rule>`.

To rerun one step on purpose: `--forcerun <rule>`.

## Running the Snakemake controller

The Snakemake controller process has to stay alive for the whole workflow. It's light, but it runs for a long time. Options:
- **tmux on a fixed login node** (`discovery1.usc.edu`). CARC documents tmux on login nodes (`module load gcc/13.3.0 tmux/3.4`), but login nodes reboot during maintenance. That's survivable, because Snakemake reruns only missing steps. [CARC]
- **Inside a long CPU job** on `main` (48 h) or `oneweek` (168 h). It's never killed by a login-node cleanup, but it costs SUs while it waits.
- **Not inside a debug or other `salloc` allocation.** The controller dies when the allocation ends, and the SLURM plugin then cancels the jobs it submitted. Snakemake warns about this when it starts inside a SLURM job context; job 13648542 was cancelled for this reason (2026-11-01).
- **In use:** tmux on `discovery1`. The L40S smoke test (job 13649053) ran this way. **Open:** check that the SLURM plugin's status polling stays well under the login node's 64-process limit.

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
