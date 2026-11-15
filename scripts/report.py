#!/usr/bin/env python3
"""Prints the benchmark results as Markdown tables.

    scripts/report.py

Reads results/known*.jsonl, results/film.jsonl and results/real.jsonl, whichever exist.
Needs only the standard library.

known: averages over the clips for every denoiser and grain profile.
film:  averages over the clips for every denoiser and CRF: bitrate against the direct
       encode, and how much texture is in what the denoiser removed.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    path = ROOT / "results" / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def known(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["profile"], r["denoiser"])].append(r)
    out = []
    for (profile, denoiser), rs in sorted(groups.items()):
        out.append([
            profile,
            denoiser,
            str(len(rs)),
            f"{mean(r['psnr_noisy']['y'] for r in rs):.2f} → {mean(r['psnr']['y'] for r in rs):.2f}",
            f"{mean(r['psnr']['u'] for r in rs):.2f} / {mean(r['psnr']['v'] for r in rs):.2f}",
            f"{mean(r['grain_removed']['y'] for r in rs):.3f}",
            f"{mean(r['grain_removed']['u'] for r in rs):.3f} / {mean(r['grain_removed']['v'] for r in rs):.3f}",
            f"{mean(r['collateral_psnr']['y'] for r in rs):.2f}",
            f"{mean(r['ssimulacra2']['mean'] for r in rs):.2f}",
            f"{mean(r['fps'] for r in rs):.1f}",
        ])
    header = ["Grain", "Denoiser", "Clips", "PSNR Y noisy → out", "PSNR U / V", "Grain removed Y",
              "Removed U / V", "Collateral PSNR Y", "SSIMULACRA2", "fps"]
    return table(header, out)


def film(rows):
    direct = {(r["clip"], r["crf"]): r["kbps"] for r in rows if r["denoiser"] == "none"}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["denoiser"], r["crf"])].append(r)
    out = []
    for (denoiser, crf), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "none", kv[0])):
        share = [r["kbps"] / direct[(r["clip"], crf)] for r in rs if (r["clip"], crf) in direct]
        residual = [r["residual"]["texture_ratio"] for r in rs if r.get("residual")]
        out.append([
            denoiser,
            str(crf),
            str(len(rs)),
            f"{mean(r['kbps'] for r in rs):.0f}",
            f"{mean(share) * 100:.0f}%",
            f"{mean(residual):.2f}" if residual else "",
            f"{mean(r['ssimulacra2']['mean'] for r in rs):.2f}",
            f"{mean(r['denoise_fps'] for r in rs):.1f}" if denoiser != "none" else "",
        ])
    header = ["Denoiser", "CRF", "Clips", "kbps", "Bitrate vs direct", "Texture ratio", "SSIMULACRA2 vs source", "Denoise fps"]
    notes = (
        "\n\nBitrate vs direct is the mean, over clips, of the encode's bitrate over the direct encode's at the same CRF. "
        "Texture ratio is residual variance in textured over flat blocks: the true grain gives about 1.16 on the known clips, "
        "a sigma 1 blur 1.5 to 1.9. SSIMULACRA2 compares the regrained decode with the source, where the random grain "
        "differs pixel by pixel, so it mostly measures grain mismatch and does not rank denoisers."
    )
    return table(header, out) + notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    printed = False
    for path in sorted((ROOT / "results").glob("known*.jsonl")):
        rows = load(path.stem)
        if rows:
            cut = path.stem[len("known"):].lstrip("-")
            if cut:
                start, frames = cut.split("-")
                title = f"Known grain, {frames} frames from frame {start}"
            else:
                title = "Known grain, whole clips"
            print(f"## {title}\n")
            print(known(rows))
            print()
            printed = True
    for name, title in (("film", "Real film through the whole pipeline"), ("real", "Your clips through the whole pipeline")):
        rows = load(name)
        if rows:
            print(f"## {title}\n")
            print(film(rows))
            print()
            printed = True
    if not printed:
        print("No results yet. Run scripts/bench.sh first.")


if __name__ == "__main__":
    main()
