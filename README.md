# Chatterbox vLLM Audiobook

Turn DRM-free EPUB books into chaptered M4B audiobooks with Chatterbox TTS and vLLM. This fork adds a Gradio web interface, sentence-aware EPUB processing, batched GPU generation, resumable projects, loudness normalization, and parallel FFmpeg assembly to [randombk/chatterbox-vllm](https://github.com/randombk/chatterbox-vllm).

The web interface currently targets English narration and defaults to the pinned **Chatterbox Multilingual V3 model in English mode**. The original English and Multilingual V2 checkpoints remain selectable through environment variables for compatibility with older projects.

> [!IMPORTANT]
> This project requires Linux or WSL2 and an NVIDIA CUDA GPU. Native Windows execution is not supported. It reads ordinary DRM-free EPUB files; it does not remove DRM.

## Features

- EPUB upload with title, author, cover, spine order, and chapter extraction
- Sentence-aware text chunking and batched vLLM speech generation
- Multilingual V3 in English mode as the default model
- A text-sample tab for testing a reference voice before starting a book
- Progress reporting with generated chunks, realtime speed, and ETA
- Background WAV saving and normalization while GPU generation continues
- EBU R128 speech normalization to -18 LUFS for previews and audiobook chunks
- V3 internal pauses capped at 500 ms without changing edge silence
- Safe stop and resume with the source EPUB, reference audio, settings, and validated WAVs retained
- Assembly-only resume when speech generation is already complete
- Chaptered AAC/M4B output with EPUB metadata and cover artwork
- Parallel FFmpeg encoding pinned to one logical CPU per detected physical core
- Final M4B verification before intermediate WAV chunks are removed

## Requirements

- x86-64 Linux or WSL2
- An NVIDIA GPU supported by the pinned PyTorch and vLLM versions
- A working NVIDIA driver; <code>nvidia-smi</code> must succeed inside Linux
- Internet access for installation and the first model download
- At least 20 GB of free disk space for the environment, CUDA libraries, model cache, and outputs
- Enough system RAM and swap for long books

The default batch size of 16 was tested on an RTX 4090 with 24 GB of VRAM. Smaller GPUs may need a lower batch size. The installer pins Python 3.12, PyTorch 2.7.1, and vLLM 0.10.0 through the committed lockfile.

The project does not require a separately installed CUDA Toolkit. PyTorch's Linux package supplies its CUDA runtime dependencies. The NVIDIA driver remains a system prerequisite and is not installed by this repository.

## WSL2 preparation

On Windows, install or update WSL from an Administrator PowerShell:

~~~powershell
wsl --install -d Ubuntu
wsl --update
wsl --list --verbose
~~~

Install a current NVIDIA Windows driver with WSL support. Do **not** install a Linux NVIDIA display driver inside WSL; WSL uses the driver supplied by Windows. Start Ubuntu and verify GPU access:

~~~bash
nvidia-smi
~~~

Do not continue until that command displays the NVIDIA GPU without an error.

## Installation

### Recommended installation

Run these commands inside Linux or Ubuntu on WSL2:

~~~bash
sudo apt update
sudo apt install -y git

git clone https://github.com/anethema/chatterbox-vllm-audiobook.git
cd chatterbox-vllm-audiobook
./install_linux.sh
~~~

The installer:

1. Confirms that it is running on Linux and that <code>nvidia-smi</code> works.
2. Installs curl and FFmpeg with APT when they are missing.
3. Installs [uv](https://docs.astral.sh/uv/) with Astral's official installer when needed.
4. Creates <code>.venv</code> with Python 3.12.
5. Installs the exact dependencies in <code>uv.lock</code>.
6. Verifies CUDA through PyTorch and prints the PyTorch, CUDA runtime, vLLM, GPU, FFmpeg, and FFprobe versions.

It does not install or modify the NVIDIA driver.

### Manual installation

~~~bash
sudo apt update
sudo apt install -y git curl ffmpeg

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/anethema/chatterbox-vllm-audiobook.git
cd chatterbox-vllm-audiobook
uv sync --locked --python 3.12
~~~

Verify the environment:

~~~bash
.venv/bin/python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
ffmpeg -version
ffprobe -version
~~~

## Running the interface

~~~bash
./run_chatterbox_vllm.sh
~~~

Open <http://127.0.0.1:7860>. The server listens on <code>0.0.0.0:7860</code>, so another device on the same trusted network can use:

~~~text
http://LINUX_MACHINE_IP:7860
~~~

On WSL2, Windows can normally open the localhost URL directly. Files in the repository are also available from Windows Explorer at a path similar to:

~~~text
\\wsl.localhost\Ubuntu\home\YOUR_UBUNTU_USER\chatterbox-vllm-audiobook
~~~

The first launch downloads the pinned model files from [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox). Multilingual V3 uses the pinned <code>t3_mtl23ls_v3.safetensors</code> checkpoint plus the shared voice encoder, S3Gen, tokenizer, and conditioning files. Hugging Face normally caches them under <code>~/.cache/huggingface/hub</code>, so later launches reuse the download.

Keep the terminal process running while generating. Closing a browser tab does not unload the model or necessarily cancel a queued server job, but reopening the page does not reconnect to the old progress display. If the process itself stops, resume the saved project after restarting it.

### Temporary public link

~~~bash
./run_chatterbox_vllm.sh --share
~~~

Gradio prints a temporary <code>gradio.live</code> URL. There is no authentication: anyone with the URL can operate the interface and access generated files exposed by it. Stop the process to close the link.

## Creating an audiobook

1. Open the **EPUB Audiobook** tab.
2. Upload or record a clean reference voice sample.
3. Upload a DRM-free EPUB and review the detected chapter and chunk counts.
4. Adjust generation settings if needed.
5. Click **Generate EPUB Audiobook**.
6. Leave the server process running through both speech generation and M4B assembly.

Current UI defaults:

| Setting | Default |
| --- | ---: |
| Model | Multilingual V3, English mode |
| Exaggeration | 0.5 |
| CFG/Pace | 0.5 |
| Diffusion steps | 15 |
| Temperature | 0.8 |
| Min-P | 0.05 |
| Top-P | 1.0 |
| Repetition penalty | 1.2 |
| Maximum chunk length | 280 characters |
| vLLM batch size | 16 |
| Loudness target | -18 LUFS |
| V3 maximum internal pause | 500 ms |

Larger batches can improve GPU throughput but increase peak resource use. Diffusion steps primarily trade speed for waveform quality. Lower CFG/Pace values loosen text guidance and often produce slower delivery, while higher exaggeration increases expressiveness and can also increase pace. Extreme sampling or exaggeration settings can reduce stability. New projects save CFG/Pace with their other generation settings so resumed chunks keep the same delivery. V3 pause limiting runs with WAV saving and normalization in the bounded CPU background pool, allowing GPU generation to continue with the next batch.

Opening another browser tab does not load another model copy; every tab uses the same server process and model. Do not start overlapping text and EPUB jobs, because separate Gradio events can contend for the same GPU, CPU, and RAM and may make a long conversion appear stalled.

Uploaded voice references are normalized from a temporary copy to -20 LUFS with a -3 dBTP ceiling before voice conditioning. The Gradio reference player is replaced with that normalized copy, while the original reference file is not changed. The two-pass linear normalization preserves dynamics when the target can be reached without exceeding the peak ceiling.

## Output files

Each conversion creates a timestamped folder beneath <code>audiobook_outputs/</code>:

~~~text
audiobook_outputs/Book-YYYYMMDD-HHMMSS-ID/
├── inputs/
│   ├── source.epub
│   └── reference-audio.ext
├── metadata.json
├── progress.json          # incomplete projects only
├── chunks/                # incomplete projects only
└── audiobook.m4b          # completed projects
~~~

The finished M4B contains AAC audio, chapter markers, and available EPUB title, author, cover, language, publisher, description, date, and identifier metadata.

After assembly, the app uses FFprobe to verify that the M4B contains readable AAC audio with a positive duration. Only after verification succeeds are <code>chunks/</code> and <code>progress.json</code> removed. The saved EPUB, reference voice, metadata, and final M4B remain. The web interface serves the verified M4B directly from this folder instead of duplicating large audiobooks in temporary storage.

## Stopping and resuming

The **Stop Generation** button requests an orderly stop. Valid normalized WAVs, project metadata, the source EPUB, and the reference voice are preserved.

To resume:

1. Restart the app with the same model used by the project.
2. Expand **Resume an incomplete project**.
3. Click **Refresh Incomplete Projects** if necessary.
4. Select the project; its saved EPUB and reference audio load automatically.
5. Click **Resume Selected Project**.

Resume validates the saved EPUB's chunk plan and scans the actual WAV files rather than trusting the displayed percentage. It restarts at a safe batch boundary if the final batch was interrupted. When every WAV is already valid, resume skips speech generation and goes directly to M4B assembly.

Projects created before saved inputs were implemented may request the original EPUB and reference audio once. A project with unfinished speech must use its recorded model version so one audiobook cannot silently mix voices from different models.

## Model selection

Multilingual V3 is the normal default for both the EPUB and text-sample tabs:

~~~bash
./run_chatterbox_vllm.sh
~~~

Compatibility variants can be selected before launch:

~~~bash
CHATTERBOX_MODEL_VARIANT=english-v1 ./run_chatterbox_vllm.sh
CHATTERBOX_MODEL_VARIANT=multilingual-v2 ./run_chatterbox_vllm.sh
CHATTERBOX_MODEL_VARIANT=multilingual-v3 ./run_chatterbox_vllm.sh
~~~

The active model is shown at the top of the page and stored in new project metadata. Although V3 uses the multilingual model and tokenizer, this audiobook UI currently invokes it in English mode and does not expose a language selector.

## M4B encoding workers

By default, assembly detects the CPUs available to the process, selects one logical CPU from each physical core, starts that many FFmpeg encoders, and pins one encoder to each selected CPU when Linux permits it.

Override the worker count when needed:

~~~bash
CHATTERBOX_M4B_WORKERS=8 ./run_chatterbox_vllm.sh
~~~

CPU-affinity tools such as Process Lasso, container CPU limits, or task affinity inherited by WSL can restrict which cores the app detects.

## Updating

The public release branch is <code>master</code>:

~~~bash
cd ~/chatterbox-vllm-audiobook
git switch master
git pull --ff-only origin master
./install_linux.sh
~~~

Rerunning the installer is safe and synchronizes <code>.venv</code> to the updated lockfile. Model files already present in the Hugging Face cache are reused.

Development changes are prepared on feature branches and merged into <code>master</code> after testing.

## Tests

Run the unit suite without loading the model:

~~~bash
.venv/bin/python -m unittest discover -v tests
~~~

The first real generation is the practical CUDA/model-load check. Start with a short text sample before converting a long book.

## Troubleshooting

### <code>nvidia-smi</code> fails

Repair or update the host NVIDIA driver first. On WSL2, update WSL with <code>wsl --update</code> from Windows and do not install a Linux display driver inside the distribution.

### PyTorch reports that CUDA is unavailable

Confirm <code>nvidia-smi</code>, then rerun <code>./install_linux.sh</code>. Do not independently upgrade PyTorch or vLLM; this repository relies on the versions in <code>uv.lock</code>.

### Port 7860 is already in use

Another copy of the app is probably running. Stop the older process before launching a replacement:

~~~bash
ss -ltnp | grep ':7860'
~~~

### A long conversion stops or reports an error

Do not delete its project folder. Restart the app, refresh the incomplete-project list, and resume it. The app preserves validated WAVs on ordinary errors, requested stops, and low-memory pauses.

### M4B assembly fails

Confirm both tools are available:

~~~bash
ffmpeg -version
ffprobe -version
~~~

The intermediate WAVs remain available when final encoding or verification fails, allowing assembly-only resume after the problem is corrected.

### Lower-memory GPU or system

Try a smaller vLLM batch size and ensure Linux has adequate RAM and swap. Only run one app process; separate processes each load their own model and compete for VRAM.

## Scope and upstream

This remains a personal project built on:

- [ResembleAI/chatterbox](https://github.com/resemble-ai/chatterbox)
- [randombk/chatterbox-vllm](https://github.com/randombk/chatterbox-vllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Gradio](https://www.gradio.app/)

The vLLM integration uses internal APIs and is intentionally pinned to vLLM 0.10.0. APIs and model behavior may change on future upgrades.

This project is not affiliated with the maintainer's employer or any other corporate entity. See [LICENSE](LICENSE) for licensing information.
