# Gate G1 substitute (written when Docker was not installed on the audit machine; still useful without a container): reproduce what `docker compose up`
# does - from a FRESH copy of the working tree (uncommitted files included, environments / caches / audio
# excluded, exactly what a clean clone of a committed tree would lack) with a brand-new venv from
# requirements.lock, a fresh Hugging Face cache (real download of the pinned models), the fixture corpus
# rebuilt from its seed and compared with the committed one, and the image's own command.
# The CPU path is used on purpose (the image is CPU-only): STREAMING_RAG_ORT_PROVIDER=cpu, the explicit opt-out.
#
# NOT covered: the Docker build itself (base image, non-root user, layer caching) and the SQuAD2 reader export
# (needs torch + transformers, Dockerfile stage 1) - the SHA-256-verified exported artefact is copied instead.
$ErrorActionPreference = "Continue"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$cr   = Join-Path $env:TEMP "cleanroom_srag"
$log  = Join-Path $repo "eval\results\final_audit\cleanroom.log"
$py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
function Log($m) { $line = "{0} {1}" -f (Get-Date -Format HH:mm:ss), $m; Add-Content -Path $log -Value $line; Write-Host $line }
Set-Content -Path $log -Value "clean-room G1 substitute"
Remove-Item -Recurse -Force $cr -ErrorAction SilentlyContinue

Log "1. copy the working tree to $cr (no .git, venvs, caches, audio, raw data)"
robocopy $repo $cr /E /XD .git .venv .venv-gpu .cache .pytest_cache __pycache__ "$repo\handoff\asr_tools\heysquad_audio" "$repo\handoff\asr_tools\whisper" "$repo\handoff\newmachine" "$repo\data\raw" /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
Set-Location $cr

Log "2. fresh venv + pip install -r requirements.lock"
$t = Get-Date
& $py311 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements.lock 2>&1 | Select-Object -Last 3 | ForEach-Object { Log "   pip: $_" }
Log ("   install took {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)
Log "3. pip check"
& .\.venv\Scripts\python.exe -m pip check 2>&1 | ForEach-Object { Log "   $_" }
& .\.venv\Scripts\python.exe -c "import sys, streaming_rag; print('import streaming_rag ok on Python', sys.version.split()[0])" 2>&1 | ForEach-Object { Log "   $_" }

Log "4. fresh HF cache: fetch + SHA-256 verify the pinned models"
$env:HF_HOME = "$cr\.hf"
$env:STREAMING_RAG_CACHE = "$cr\.cache\streaming_rag"
$env:STREAMING_RAG_ORT_PROVIDER = "cpu"; $env:STREAMING_RAG_ASR_DEVICE = "cpu"
& .\.venv\Scripts\python.exe -m streaming_rag.retrieval.neural --fetch 2>&1 | Select-Object -Last 2 | ForEach-Object { Log "   $_" }
& .\.venv\Scripts\python.exe -m streaming_rag.retrieval.neural --verify 2>&1 | ForEach-Object { Log "   $_" }

Log "5. SQuAD2 reader ONNX: copy the exported artefact and compare its hash with the recorded one"
$src = Join-Path $repo ".cache\streaming_rag\models\minilm-uncased-squad2-onnx"
$dst = Join-Path $cr ".cache\streaming_rag\models\minilm-uncased-squad2-onnx"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src\*" $dst -Recurse -Force
$want = "6e9e62e6c6acb891cca9896e6f6d2f939c28e4b810e885b902372dfe340bd421"
$got = (Get-FileHash "$dst\model.onnx" -Algorithm SHA256).Hash.ToLower()
Log ("   reader sha256 = $got")
Log ("   matches the recorded export hash: " + ($got -eq $want))

Log "6. rebuild the fixture corpus from its seed and diff against the committed one"
& .\.venv\Scripts\python.exe -m fixtures.build_dev_corpus --seed 1 --out "$cr\_rebuilt_dev_corpus" 2>&1 | Select-Object -Last 1 | ForEach-Object { Log "   $_" }
$a = Get-ChildItem "$cr\fixtures\dev_corpus" -Recurse -File | Sort-Object FullName
$b = Get-ChildItem "$cr\_rebuilt_dev_corpus" -Recurse -File | Sort-Object FullName
$diff = 0
foreach ($f in $a) { $rel = $f.FullName.Substring("$cr\fixtures\dev_corpus".Length); $g = "$cr\_rebuilt_dev_corpus$rel";
  if (-not (Test-Path $g) -or (Get-FileHash $f.FullName).Hash -ne (Get-FileHash $g).Hash) { $diff++; Log "   DIFFERS: $rel" } }
Log ("   files committed={0} rebuilt={1} differing/missing={2}" -f $a.Count, $b.Count, $diff)

Log "7. the image's command: python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1 (CPU)"
$t = Get-Date
& .\.venv\Scripts\python.exe -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1 2>&1 | ForEach-Object { Log "   $_" }
Log ("   run_all wall time {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)

Log "8. pytest in the clean CPU environment"
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider 2>&1 | Select-Object -Last 3 | ForEach-Object { Log "   $_" }
Log "DONE"
