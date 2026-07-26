#!/usr/bin/env bash
# Provider-free, non-live usability gate.  All writes stay below OUTPUT_DIR.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?usage: provider_free_usability_gate.sh OUTPUT_DIR}"
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output directory: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
export PYTHONPATH="$repo_root/src"

python3 - "$repo_root" "$output_dir" <<'PY'
import re, sys
from pathlib import Path
from petta_memory.store import MediumMemoryStore

repo, output = map(Path, sys.argv[1:])
fixture = (repo / "fixtures/e2e_journal.metta").read_text(encoding="utf-8")
records = re.findall(r";;; BEGIN MemoryCluster [^\n]+\n(.*?);;; END MemoryCluster [^\n]+", fixture, flags=re.S)
if not records:
    raise SystemExit("fixture contains no complete MemoryCluster records")
store = MediumMemoryStore(output / "journal.metta")
for record in records:
    store.append_cluster(record)
if len(store.clusters()) != len(records):
    raise SystemExit("ingested cluster count does not match fixture")
PY

sha256sum "$output_dir/journal.metta" > "$output_dir/journal.after-ingest.sha256"
python3 -m petta_memory.cli --store "$output_dir/journal.metta" index-view > "$output_dir/index.metta"
python3 -m petta_memory.cli --store "$output_dir/journal.metta" query about MediumPeTTaMemory > "$output_dir/retrieval.metta"
grep -q 'MM-index-about MediumPeTTaMemory' "$output_dir/index.metta"
grep -q 'mc-e2e-commitment' "$output_dir/retrieval.metta"
grep -q 'mc-e2e-promotion' "$output_dir/index.metta"

# Local patham9/PLN inference: isolated temp program, fixed 30-second bound.
python3 -m petta_memory.cli --store "$output_dir/journal.metta" \
  patham9-pln-derivation-smoke --pln-repo "$repo_root/../patham9-pln" --timeout-sec 30 \
  > "$output_dir/inference.json"
python3 - "$output_dir/inference.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"local derivation was not semantically passed: {result.get('classification')}")
PY

# Independent process restart/retrieval and an explicitly enabled read-only canary.
python3 -m petta_memory.cli --store "$output_dir/journal.metta" query about MediumPeTTaMemory > "$output_dir/retrieval.after-restart.metta"
cmp "$output_dir/retrieval.metta" "$output_dir/retrieval.after-restart.metta"
python3 - "$output_dir/journal.metta" "$output_dir/read_only_canary.metta" <<'PY'
import hashlib, sys
from pathlib import Path
from petta_memory.omegaclaw import OmegaClawMemoryBridge, OmegaClawMemoryPolicy
from petta_memory.store import MediumMemoryStore

journal, canary = map(Path, sys.argv[1:])
before = hashlib.sha256(journal.read_bytes()).hexdigest()
bridge = OmegaClawMemoryBridge(MediumMemoryStore(journal), OmegaClawMemoryPolicy(
    prompt_view_reads_enabled=True, index_view_reads_enabled=True,
    prompt_topics=frozenset({"MediumPeTTaMemory"}), prompt_statuses=frozenset({"active"}),
    view_id="private-readonly-canary", index_view_id="private-readonly-index",
))
prompt = bridge.prompt_view_metta(generated_at="2026-07-26T00:00:00Z")
index = bridge.index_view_metta(generated_at="2026-07-26T00:00:00Z")
after = hashlib.sha256(journal.read_bytes()).hexdigest()
if before != after:
    raise SystemExit("read-only canary changed the journal")
if "OmegaClawPromptView" not in prompt or "OmegaClawIndexView" not in index:
    raise SystemExit("read-only canary emitted incomplete views")
canary.write_text(prompt + "\n" + index, encoding="utf-8")
PY
sha256sum "$output_dir/journal.metta" > "$output_dir/journal.after-canary.sha256"
cmp "$output_dir/journal.after-ingest.sha256" "$output_dir/journal.after-canary.sha256"

python3 - "$output_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
names = ["journal.metta", "index.metta", "retrieval.metta", "inference.json", "read_only_canary.metta"]
summary = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
summary["journal_unchanged_by_canary"] = True
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
