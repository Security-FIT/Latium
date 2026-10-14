"""Small command surface for CSF reproduction and saved-artifact rendering."""
import argparse
import json
from pathlib import Path

from src.submission.inputs import CAMPAIGNS, load_campaign, select_entries


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def selection_options(parser):
    parser.add_argument("campaign", choices=CAMPAIGNS)
    parser.add_argument("--family")
    parser.add_argument("--checkpoint", help="Exact frozen repository ID or model key")
    parser.add_argument("--limit-checkpoints", type=positive, help="Use the first selected checkpoints; marks a validation subset")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen CSF campaigns, ordinary Linux/SSH execution and read-only artifact exports")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Inspect the fixed selection without downloading models")
    selection_options(plan)
    for name in ("run", "smoke"):
        command = commands.add_parser(name, help="Execute frozen inputs" if name == "run" else "One-fact validation with explicitly reduced covariance")
        selection_options(command)
        command.add_argument("--output", required=True)
        command.add_argument("--base-revisions", help="JSON repository-ID -> original commit SHA for unrecorded classic/covariance base revisions")
        command.add_argument("--base-covariance-dir", help="Original 100,000-sample base tensors, named as in the saved configs")
        command.add_argument("--limit-facts", type=positive)
        command.add_argument("--prefix-mode", choices=("self", "archived"), default="self",
                             help="Self-generated contexts by default; archived retains the original recorded prefix protocol for comparisons")
        if name == "smoke":
            command.add_argument("--covariance-samples", type=positive, default=32)
    render = commands.add_parser("render", help="Export an archived campaign to a separate directory")
    render.add_argument("campaign", choices=CAMPAIGNS)
    render.add_argument("--archive", required=True)
    render.add_argument("--output", required=True)
    new = commands.add_parser("render-new", help="Aggregate newly reproduced campaign runs, including the original repair selection")
    new.add_argument("campaign", choices=CAMPAIGNS)
    new.add_argument("--runs", required=True)
    new.add_argument("--output", required=True)
    single = commands.add_parser("render-run", help="Export one saved run without rewriting its manifest")
    single.add_argument("run_root")
    single.add_argument("--output", required=True)
    validate = commands.add_parser("validate", help="Verify frozen inputs or a downloaded archive")
    validate.add_argument("--archive")
    fetch = commands.add_parser("download-archive", help="Download the four published folders read-only")
    fetch.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = load_campaign(args.campaign)
        entries = select_entries(payload, family=args.family, checkpoint=args.checkpoint, limit=args.limit_checkpoints)
        result = {"campaign": args.campaign, "frozen_entries": len(payload["entries"]), "selected_entries": len(entries),
                  "entries": [{"family": e["family"], "model_id": e["model_id"], "revision": e.get("revision"),
                               "evaluated": e.get("evaluated", True), "fact_position": e.get("fact_position"),
                               "fact_case_id": e.get("fact_case_id"), "execution_seed": e.get("execution_seed"),
                               "batches": [{k: b[k] for k in ("name", "seed", "start", "count")} for b in e.get("batches", [])]}
                              for e in entries]}
    elif args.command in {"run", "smoke"}:
        from src.submission.run import execute
        result = execute(args, smoke=args.command == "smoke")
    elif args.command == "render":
        from src.submission.render import render_campaign
        result = render_campaign(args.campaign, args.archive, args.output)
    elif args.command == "render-run":
        from src.submission.render import render_saved_run
        result = render_saved_run(args.run_root, args.output)
    elif args.command == "render-new":
        from src.submission.render import render_new_campaign
        result = render_new_campaign(args.campaign, args.runs, args.output)
    elif args.command == "download-archive":
        from src.submission.archive import download_archives
        result = download_archives(args.output)
    elif args.archive:
        from src.submission.archive import verify_archive
        result = verify_archive(args.archive)
        if not result["ok"]:
            print(json.dumps(result, indent=2))
            return 1
    else:
        from src.submission.inputs import verify_packaged_inputs
        result = verify_packaged_inputs()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
