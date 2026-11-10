# CUE: Chemical-Use Extraction

CUE is a local web application for extracting chemical-use relationships from
text, CSV, TSV, and Excel files. It uses one merged extraction model and can
optionally add OECD functional-use labels and chemical database identifiers.

## Start The App

### Windows Release

Windows users can download
[Chemical Use Extractor v1.0](https://github.com/xinghd142857/CUE-ChemicalUseExtraction/releases/download/v1.0/Chemical.Use.Extractor.v1.0.exe).

> **Important:** Before opening `Chemical.Use.Extractor.exe`, place it in the
> folder where you want CUE to store its downloaded models and runtime files.
> The Llama-CUE model download requires approximately 16 GB of free storage.

Double-click the file to start CUE. Keep the application window open while
using the web interface.

### Run From Source

Clone or download this repository, open a terminal in the project folder, and
run the commands for your operating system.

#### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_windows.ps1
```

#### macOS Or Linux

```bash
bash scripts/setup_unix.sh
bash scripts/run_unix.sh
```

Open [http://127.0.0.1:8888](http://127.0.0.1:8888) in a browser. Keep the
terminal open while using CUE. The setup command only needs to be run again
when the environment or dependencies change.

## Extraction Model

CUE uses the merged model from `Jesolo/Llama-CUE`. The model is not stored in
this repository.

After starting the app, click **Download Llama-CUE model** in the web interface.
The progress bar shows the download status. The default model folder is
`models/Llama-CUE`.

The relevant defaults in `config.example.json` are:

```json
{
  "model_path": "models/Llama-CUE",
  "model_repo_id": "Jesolo/Llama-CUE",
  "model_revision": "main",
  "model_download_dir": "models/Llama-CUE",
  "extraction_backend": "transformers",
  "device": "auto",
  "port": 8888
}
```

The setup script creates `config.local.json` automatically. Edit that local
file to override settings. Relative paths are resolved from the project root.
For a private Hugging Face repository, set `HF_TOKEN` in the terminal before
starting CUE.

## Start Ollama For OECD Labels

OECD functional-use labeling is optional. Install and start
[Ollama](https://ollama.com/download), then download and run the configured
model:

```bash
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```

Keep Ollama running while using OECD labels in CUE. The default connection is:

```json
{
  "oecd_provider": "ollama",
  "oecd_ollama_base_url": "http://127.0.0.1:11434/v1",
  "oecd_ollama_model": "gpt-oss:20b"
}
```

The Ollama model is separate from Llama-CUE. Ollama lists
[`gpt-oss:20b`](https://ollama.com/library/gpt-oss:20b) at approximately 14 GB.

## Use CUE

- Paste a paragraph for direct extraction.
- Upload CSV, TSV, XLSX, or XLS files and choose the input column.
- Export completed results as CSV, JSON, or XLSX.
- Use the **Train** tab to prepare LoRA or DPO settings, validate training-data
  columns, and estimate memory from the detected machine information.

Generated results are stored under `runs/`, downloaded models under `models/`,
and local caches under `cache/`. These runtime folders are ignored by Git.

## Chemical Enrichment

Chemical enrichment is optional and requires internet access. When enabled,
CUE looks up identifiers and structures through:

- PubChem PUG REST
- NCI CACTUS Chemical Identifier Resolver
- ChEMBL search

CUE also provides search links for EPA CompTox, CAS Common Chemistry, and
Wikidata when available. Successful results are cached for seven days in
`cache/chemical_cache.json`. An unavailable external service does not stop the
remaining extraction results.

## Notes

- Direct text input is limited to 500 words.
- File extraction supports up to 100,000 rows.
- A failed row does not stop the remaining file extraction.
- The app can start before the merged model is downloaded, but extraction will
  remain unavailable until the model folder is ready.
