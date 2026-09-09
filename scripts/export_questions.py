"""Provision question-only inputs; never decode excluded field values.

Run beside the trusted grader, not in the solver. This lexical projection copies
only the two allowed top-level fields. It never interprets answer values, emits
source rows, or includes source data in exception messages.
"""

import argparse
import json
from pathlib import Path


def value_end(text, start):
    quoted = escaped = False
    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            if depth == 0:
                return pos
            depth -= 1
        elif ch == "," and depth == 0:
            return pos
    raise ValueError("Malformed source record")


def project(text):
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if text[pos : pos + 1] != "{":
        raise ValueError("Expected an object")
    pos += 1
    result = {}
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos : pos + 1] == "}":
            break
        key, pos = decoder.raw_decode(text, pos)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos : pos + 1] != ":":
            raise ValueError("Expected a field separator")
        pos += 1
        end = value_end(text, pos)
        if key in ("problem_idx", "problem"):
            if key in result:
                raise ValueError("Duplicate input field")
            result[key] = decoder.decode(text[pos:end])
        pos = end
        if text[pos] == ",":
            pos += 1
        else:
            break
    if set(result) != {"problem_idx", "problem"}:
        raise ValueError("Missing question fields")
    if type(result["problem_idx"]) is not int or not isinstance(result["problem"], str):
        raise ValueError("Invalid question field types")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = 0
    with args.source.open(encoding="utf-8-sig") as src, args.output.open(
        "x", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.strip():
                dst.write(json.dumps(project(line), ensure_ascii=False) + "\n")
                count += 1
    print(f"Exported {count} question-only records")


if __name__ == "__main__":
    main()
