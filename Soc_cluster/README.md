## This guide is from https://github.com/isaacNingLee/Unofficial_SoC_Cluster_Guide

# Unofficial NUS SoC Compute Cluster Guide

A beginner-friendly quick-start guide to connecting to the NUS School of Computing (SoC) compute cluster, setting up Conda, finding an available GPU, and running Slurm jobs.

## Official references

Always follow the official documentation when it differs from this guide:

- [Accessing the compute cluster](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access)
- [Getting started with the compute cluster](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/start)
- [Compute cluster hardware](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware)
- [Job priority](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/job-priority) — requires an SoC account login
- [StuJump quick guide](https://dochub.comp.nus.edu.sg/cf/guides/sjump/quick_guide)

## ⚠️ Disclaimer — please read

> [!WARNING]
> **This is an unofficial guide based on my personal notes.** It is not maintained, verified, or endorsed by NUS School of Computing. Cluster configuration, storage limits, available hardware, and scheduling policies may change. These notes were polished and formatted with Codex (GPT-5.6 Sol). Always consult the official documentation above when it differs from this guide.
>
> I maintain this guide voluntarily in my free time. If you report an issue or open a pull request, please allow ample time for me to reply, as I may be busy with other commitments.

Found an error or something useful to add? Students and staff are warmly encouraged to share improvements by opening a pull request.

> [!IMPORTANT]
> **Connect via a plain terminal SSH client only.** Do not use VSCode's Remote-SSH extension (or similar IDE-integrated SSH tools) to connect to the cluster — this is not allowed.

## Prerequisites

Before continuing, complete the official [cluster access steps](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access).

You will need:

- an SoC account;
- an SSH client; and
- StuJump configured if you are connecting from outside the SoC network.

## 1. Connect to the cluster

### From an SoC building

Open a terminal and run:

```bash
ssh <soc_username>@xlogin.comp.nus.edu.sg
```

Replace `<soc_username>` with your SoC username.

### From outside SoC

First complete the [StuJump setup](https://dochub.comp.nus.edu.sg/cf/guides/sjump/quick_guide). Then connect through StuJump:

```bash
ssh -J <soc_username>@stujump.comp.nus.edu.sg <soc_username>@xlogin.comp.nus.edu.sg
```

### If you were given a dedicated GPU node instead

> [!NOTE]
> This applies only if you were given direct access to a specific named GPU machine (for example `cgpa6`, `cgpa9`, or `cgpa10`) rather than to the shared Slurm-managed compute cluster the rest of this guide describes.

Connect to that machine by hostname instead of `xlogin`:

```bash
# From an SoC building.
ssh <soc_username>@<node_name>.comp.nus.edu.sg

# From outside SoC, via StuJump.
ssh -J <soc_username>@stujump.comp.nus.edu.sg <soc_username>@<node_name>.comp.nus.edu.sg
```

These machines are standalone GPU servers, not part of the Slurm cluster. There is no job queue, so `salloc`, `srun`, `sbatch`, and `squeue` do not apply — install Conda and run your scripts directly on the node instead of following the "Submit a batch job" section below. The rest of this guide's cluster-specific steps (storage paths, job priority, batch submission) do not apply to these nodes either.

Policies for dedicated nodes, including whether IDE-integrated SSH tools are permitted, may differ from the shared cluster's. Confirm with whoever granted you access if unsure.

## 2. Start an interactive compute session

After connecting, you are on the `xlogin` login node. Use it only to prepare and submit work—not for installations, training, inference, or other compute-heavy tasks.

> [!WARNING]
> Do not install Conda or run your project directly on `xlogin`. Its temporary disk space is limited. First request a compute node using Slurm.

Run these commands one at a time:

```bash
# Ask Slurm to reserve a compute node for 120 minutes (two hours).
# Your terminal may wait here until a suitable node becomes available.
salloc -t 120

# Start an interactive Bash shell on the node reserved above.
# Run your installation and computation commands inside this shell.
srun --pty bash
```

Your command prompt may change when the compute-node shell starts. When finished, leave both layers cleanly:

```bash
# Leave the Bash shell running on the compute node.
# You return to the shell that owns the Slurm allocation.
exit

# Release the Slurm allocation and its reserved resources.
# You return to the xlogin shell.
exit
```

In short, `salloc` reserves resources, `srun` starts a task on those resources, and the two `exit` commands leave the task and allocation respectively.

### Request a specific GPU or node

Use the `gpufree` helper installed later in this guide to view current GPU availability. The static [hardware list](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware) may not reflect what is currently free.

```bash
# Example: request a particular GPU resource.
salloc -G h100-47

# Example: request a particular compute node.
salloc -w xcna99
```

These resource and node names are examples and may change. Choose an available resource reported by the cluster.

## 3. Know where your files are stored

Each user has two separate 500 GB storage locations:

| Location | Allowance | Path |
| --- | ---: | --- |
| Home directory | 500 GB | `/home/<first-letter>/<username>` |
| Scratch directory | 500 GB | `/mnt/scratch/<first-letter>/<username>` |

For example, if your username is `e0123456`, the paths are:

```text
/home/e/e0123456
/mnt/scratch/e/e0123456
```

You do not need to construct the paths manually. Bash already stores your username in `$USER`, and your home directory in `$HOME`:

```bash
# Print your username and home-directory path.
echo "$USER"
echo "$HOME"

# Print your scratch-directory path.
echo "/mnt/scratch/${USER:0:1}/$USER"

# Move to your scratch directory.
cd "/mnt/scratch/${USER:0:1}/$USER"
```

The two allowances belong to different locations; free space in one does not increase the allowance in the other. Keep an independent copy of important files and check the official cluster documentation for current backup, retention, and cleanup policies.

Each location contains a hidden `.diskusage` file that reports its disk usage. The cluster updates these files every hour, so the values may be up to one hour behind your latest file changes:

```bash
# Check usage in your 500 GB home directory.
cat "$HOME/.diskusage"

# Check usage in your 500 GB scratch directory.
cat "/mnt/scratch/${USER:0:1}/$USER/.diskusage"
```

Because files whose names begin with `.` are hidden by default, a normal `ls` may not show `.diskusage`. Use `ls -la` if you want to see it in the directory listing.

To get a live breakdown of large files and directories in your current location:

```bash
# Show the total space used by the current directory.
du -sh .

# Show the size of each visible item, sorted from smallest to largest.
du -sh -- * 2>/dev/null | sort -h
```

## 4. Install Miniconda

Do this **inside an interactive compute session**, not directly on `xlogin`.

You can review the [official Miniconda installation guide](https://www.anaconda.com/docs/getting-started/installation) before installing it. Miniconda should be installed for your own user account; administrator access is not required.

The following commands download the Linux x86-64 Miniconda installer, install it under your home directory, and enable Conda in Bash:

```bash
# Download the installer to your home directory.
cd "$HOME"
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda without interactive prompts.
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"

# Load Conda in the current shell and configure future Bash shells.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

Start a new shell, or reload your Bash configuration:

```bash
source ~/.bashrc
```

Create and activate a project environment:

```bash
# Replace myenv and the Python version if your project requires different values.
conda create -n myenv python=3.11
conda activate myenv

# Install the packages required by your project, for example:
conda install numpy pandas
```

Useful Conda commands:

```bash
# Show all environments. An asterisk marks the active environment.
conda env list

# Show packages installed in the active environment.
conda list

# Leave the active environment.
conda deactivate
```

## 5. Install the cluster helper commands

The [`bashrc_helpers.sh`](bashrc_helpers.sh) file defines:

- `gpufree`, which summarizes free and total GPUs by type; and
- `show_job`, which displays your current and recently completed Slurm jobs.

From the root of this repository, install and load the helpers with:

```bash
# Keep a copy in your home directory so it remains available outside this repository.
cp bashrc_helpers.sh "$HOME/.soc_cluster_helpers.sh"

# Add the source command only if it is not already present in ~/.bashrc.
grep -qxF 'source "$HOME/.soc_cluster_helpers.sh"' "$HOME/.bashrc" || \
  printf '\nsource "$HOME/.soc_cluster_helpers.sh"\n' >> "$HOME/.bashrc"

# Load the helpers now; future Bash shells will load them automatically.
source "$HOME/.bashrc"
```

You can then run:

```bash
# Display GPU availability by GPU type.
gpufree

# Display your running, pending, and recent completed/failed jobs.
show_job
```

Other useful cluster commands:

```bash
# Show GPU utilization, memory usage, and GPU processes on the current node.
nvidia-smi

# Show your jobs currently known to the Slurm queue.
squeue --me
```

## 6. Understand job priority

A pending job is not necessarily stuck, and the queue is not necessarily processed in simple first-in, first-out order. Before requesting resources, read the official [SoC job-priority guide](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/job-priority). You must log in with your SoC account to view that page.

Scheduling rules can change, so this unofficial guide does not reproduce them. The official page is the authoritative source for how SoC determines job priority.

Use these commands to inspect your own jobs:

```bash
# List your jobs. The final column explains why a pending job is waiting.
squeue --me -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

# Show detailed information about one job.
# Replace <job_id> with the number returned by sbatch or squeue.
scontrol show job <job_id>

# Show the job's scheduling-priority components, if enabled on the cluster.
sprio -j <job_id>

# Ask Slurm for an estimated start time, if an estimate is available.
squeue --start -j <job_id>
```

Common pending reasons appear in the final `squeue` column. For example, `Resources` means the requested resources are not currently available. Always use the official job-priority page to interpret SoC-specific scheduling behavior.

## 7. Submit a batch job

The [`job_template.sbatch`](job_template.sbatch) file is an example Slurm job script. It:

- requests one A100 40 GB GPU for six hours;
- activates a specified Conda environment;
- writes combined output and errors to `logs/slurm/`; and
- forwards additional arguments to the pipeline script.

Copy the template into your project:

```bash
cp job_template.sbatch /path/to/your/project/job.sbatch
cd /path/to/your/project
```

Edit `job.sbatch` and, at minimum, change:

- `#SBATCH --job-name` to a name for your job;
- `CONDA_ENV` to the name of your Conda environment;
- `PIPELINE_SCRIPT` to the script your job should run; and
- the `#SBATCH` resource requests to values suitable for your workload.

Submit it **from your project repository root**:

```bash
sbatch job.sbatch
```

Check the job and its log file:

```bash
show_job
ls logs/slurm
```

Submitting with extra arguments forwards them to the pipeline script:

```bash
sbatch job.sbatch --num_shards 4 --shard_id 0
```