import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval" / "results" / "ground_truth_results.jsonl"
SNAP = ROOT / "eval" / "regression_baseline.json"


def load(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["save", "check"])
    p.add_argument("results", nargs="?", default=str(BASE))
    a = p.parse_args()
    rows = load(a.results)
    current = {r["id"]: bool(r["correct"]) for r in rows}
    if a.command == "save":
        SNAP.write_text(json.dumps(current, indent=2))
        print(f"Saved {len(current)} regression cases")
    else:
        if not SNAP.exists():
            raise SystemExit(
                "No regression baseline. Run: python -m eval.regression save"
            )
        old = json.loads(SNAP.read_text())
        reg = [k for k, v in old.items() if v and not current.get(k, False)]
        print(f"Previously passing: {sum(old.values())}; regressions: {len(reg)}")
        for x in reg:
            print("REGRESSION", x)
        raise SystemExit(1 if reg else 0)


if __name__ == "__main__":
    main()
