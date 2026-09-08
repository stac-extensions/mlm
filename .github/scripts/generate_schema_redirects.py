#!/usr/bin/env python3
"""Generate 'vX' / 'vX.Y' redirect copies of the strict MLM schema for GitHub Pages.

Each release publishes the strict schema (exact-version check) under
'vX.Y.Z/schema.json'. This script additionally copies it into 'vX.Y/schema.json'
and 'vX/schema.json' so partial-version references resolve to the latest
matching release. The 'const' exact-version check is relaxed to a 'pattern'
accepting any version in that alias' range. The strict input schema itself
is only read here, never modified.

An alias is only (re)generated if the released version is the highest one
known (from existing git tags) in its bucket (major, or major.minor), so an
older backported release can't overwrite a newer alias.
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_PREFIX = "https://stac-extensions.github.io/mlm"
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def versions_from_tags() -> list[tuple[int, int, int]]:
    """All 'vX.Y.Z' tags already known to git, parsed into comparable tuples."""
    proc = subprocess.run(["git", "tag", "-l", "v*"], check=True, capture_output=True, text=True)
    tags = (TAG_RE.match(t.strip()) for t in proc.stdout.splitlines())
    return [tuple(map(int, m.groups())) for m in tags if m]


def is_latest(version: tuple, all_versions: list[tuple], bucket_len: int) -> bool:
    """True if `version` is the greatest among all versions sharing its first `bucket_len` parts."""
    bucket = [v for v in [*all_versions, version] if v[:bucket_len] == version[:bucket_len]]
    return version == max(bucket)


def redirect_schema(base: dict, alias: str, pattern: str) -> dict:
    """Deep copy of `base` re-pointed at `alias`, relaxing the exact-version check to `pattern`."""
    schema = copy.deepcopy(base)
    schema["$id"] = f"{SCHEMA_PREFIX}/{alias}/schema.json"
    contains = schema["$defs"]["stac_extensions_mlm"]["properties"]["stac_extensions"]["contains"]
    del contains["const"]
    contains.update(type="string", pattern=pattern)
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, type=Path, help="Path to the strict json-schema/schema.json.")
    parser.add_argument("--tag", required=True, help="Released version tag, e.g. 'v1.5.2'.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write redirect schemas into.")
    parser.add_argument("--github-output", type=Path, help="Path to $GITHUB_OUTPUT (defaults to stdout).")
    args = parser.parse_args()

    match = TAG_RE.match(args.tag)
    if not match:
        sys.exit(f"::error::'{args.tag}' is not a 'vX.Y.Z' tag.")
    major, minor, _patch = map(int, match.groups())
    version = (major, minor, _patch)
    all_versions = versions_from_tags()
    base_schema = json.loads(args.schema.read_text())

    # SCHEMA_PREFIX only contains '.' to escape (no other regex-special chars); avoid
    # re.escape() here since it also escapes '-', which is an invalid escape in the
    # unicode-mode regex JS validators (e.g. ajv) compile these patterns with.
    prefix = SCHEMA_PREFIX.replace(".", r"\.")
    # bucket_len=2 -> "is this the newest patch for its minor?" (drives the vX.Y alias)
    # bucket_len=1 -> "is this the newest minor.patch for its major?" (drives the vX alias)
    aliases = {
        f"v{major}.{minor}": (2, rf"^{prefix}/v{major}\.{minor}(\.[0-9]+)?/schema\.json$"),
        f"v{major}": (1, rf"^{prefix}/v{major}(\.[0-9]+){{0,2}}/schema\.json$"),
    }

    outputs = {}
    for alias, (bucket_len, pattern) in aliases.items():
        name, flag = ("minor_alias", "has_minor_alias") if bucket_len == 2 else ("major_alias", "has_major_alias")
        publish = is_latest(version, all_versions, bucket_len)
        outputs[name], outputs[flag] = alias, str(publish).lower()
        if publish:
            out_file = args.output_dir / alias / "schema.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(redirect_schema(base_schema, alias, pattern), indent=2) + "\n")
            print(f"generated {out_file}")
        else:
            print(f"skipped {alias}: a newer release already covers it")

    lines = "\n".join(f"{key}={value}" for key, value in outputs.items())
    if args.github_output:
        with args.github_output.open("a") as out:
            out.write(lines + "\n")
    else:
        print(lines)


if __name__ == "__main__":
    main()
