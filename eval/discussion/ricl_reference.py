#!/usr/bin/env python3
"""Reference RICL utility for the ICL discussion.

This file provides two pieces:
1. A lightweight, dependency-free reference implementation of the core RICL
   mechanics: retrieval, neighbor ranking, and action interpolation.
2. A paper-report mode that prints the discussion result used by the artifact.

The repo does not ship the official RICL checkpoint or demo buffer, so the
`paper-report` mode is the default behavior used by `compare_icl.sh`.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DemoExample:
    name: str
    embedding: Tuple[float, ...]
    action_chunk: Tuple[Tuple[float, ...], ...]
    metadata: dict


@dataclass(frozen=True)
class RetrievalResult:
    demo: DemoExample
    distance: float


class RiclReference:
    def __init__(self, lambda_weight: float = 10.0, num_neighbors: int = 4):
        self.lambda_weight = float(lambda_weight)
        self.num_neighbors = int(num_neighbors)

    @staticmethod
    def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError("embedding dimensionality mismatch")
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def softmax(logits: Sequence[float]) -> List[float]:
        if not logits:
            return []
        peak = max(logits)
        exps = [math.exp(x - peak) for x in logits]
        denom = sum(exps)
        if denom == 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [value / denom for value in exps]

    def retrieve(self, query_embedding: Sequence[float], demos: Sequence[DemoExample]) -> List[RetrievalResult]:
        ranked = [
            RetrievalResult(demo=demo, distance=self.l2_distance(query_embedding, demo.embedding))
            for demo in demos
        ]
        ranked.sort(key=lambda item: item.distance)
        return ranked[: self.num_neighbors]

    def interpolate_action_chunks(
        self,
        query_action_chunk: Sequence[Sequence[float]],
        retrieved: Sequence[RetrievalResult],
    ) -> List[List[float]]:
        if not query_action_chunk:
            return []
        if not retrieved:
            return [list(map(float, step)) for step in query_action_chunk]

        raw_neighbor_scores = [math.exp(-self.lambda_weight * item.distance) for item in retrieved]
        total_score = 1.0 + sum(raw_neighbor_scores)
        query_weight = 1.0 / total_score
        weights = [score / total_score for score in raw_neighbor_scores]

        output: List[List[float]] = []
        step_count = len(query_action_chunk)
        step_dim = len(query_action_chunk[0])

        for step_idx in range(step_count):
            blended = [0.0] * step_dim
            query_step = query_action_chunk[step_idx]
            for dim_idx, value in enumerate(query_step):
                blended[dim_idx] += query_weight * float(value)

            for weight, item in zip(weights, retrieved):
                demo_chunk = item.demo.action_chunk
                if step_idx >= len(demo_chunk):
                    continue
                demo_step = demo_chunk[step_idx]
                if len(demo_step) != step_dim:
                    raise ValueError("action dimensionality mismatch")
                for dim_idx, value in enumerate(demo_step):
                    blended[dim_idx] += weight * float(value)

            output.append(blended)

        return output


def load_octo_task_sequence(script_path: Path) -> List[str]:
    text = script_path.read_text(encoding="utf-8")
    match = re.search(r'OCTO_ENVS_ID="\$\{OCTO_ENVS_ID:-([^}]*)\}"', text)
    if not match:
        raise ValueError(f"failed to locate OCTO_ENVS_ID in {script_path}")
    return list(ast.literal_eval(match.group(1)))


def load_demo_bank(path: Path) -> List[DemoExample]:
    demos: List[DemoExample] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        payload = json.loads(raw_line)
        demos.append(
            DemoExample(
                name=str(payload.get("name", f"demo_{line_no}")),
                embedding=tuple(float(x) for x in payload["embedding"]),
                action_chunk=tuple(tuple(float(y) for y in step) for step in payload["action_chunk"]),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return demos


def build_placeholder_demo_bank(task_sequence: Sequence[str]) -> List[DemoExample]:
    demos: List[DemoExample] = []
    for index, task in enumerate(task_sequence):
        embedding = tuple(float((sum(ord(ch) for ch in task) + index + offset) % 97) / 97.0 for offset in range(8))
        action_chunk = tuple(
            tuple(float(((index + 1) * (step_idx + 1) + dim_idx) % 13) / 13.0 for dim_idx in range(6))
            for step_idx in range(15)
        )
        demos.append(
            DemoExample(
                name=task,
                embedding=embedding,
                action_chunk=action_chunk,
                metadata={"task": task, "index": index},
            )
        )
    return demos


def print_paper_report(task_sequence: Sequence[str], first_task: str, fifth_task: str) -> None:
    print("[ICL Discussion]")
    print("Workload: Single-arm robot (Octo)")
    print("Reference workload script: eval/acc_comparison/run_acc_task_env_change.sh")
    print()
    print(f"Task sequence used by the single-arm workload contains {len(task_sequence)} task/environment settings.")
    print(f"- First new task: {first_task}")
    print(f"- Fifth new task: {fifth_task}")
    print()
    print("Paper-reported comparison with RICL:")
    print("- On the first new task, RICL achieves 38.13% lower accuracy than VLASelect.")
    print("- On the fifth new task, RICL achieves 49.74% lower accuracy than VLASelect.")
    print()
    print("Interpretation:")
    print("- The paper uses this result to show that ICL struggles for VLA models in dynamic environments.")
    print("- VLA models must output action sequences under task and environment changes, so updating only non-parametric context is not enough.")
    print()
    print("Artifact note:")
    print("- This repository does not include a runnable RICL artifact or checkpoint.")
    print("- This repository also includes a reference implementation of RICL-style retrieval and interpolation.")


def run_reference_demo(task_sequence: Sequence[str], num_neighbors: int, lambda_weight: float) -> None:
    model = RiclReference(lambda_weight=lambda_weight, num_neighbors=num_neighbors)
    demos = build_placeholder_demo_bank(task_sequence)
    query = demos[0]
    retrieved = model.retrieve(query.embedding, demos[1:])
    blended = model.interpolate_action_chunks(query.action_chunk, retrieved)

    print("[RICL Reference]")
    print(f"num_neighbors={num_neighbors}")
    print(f"lambda_weight={lambda_weight:.2f}")
    print("retrieved:")
    for item in retrieved:
        print(f"- {item.demo.name}: distance={item.distance:.6f}")
    print("first blended action step:")
    if blended:
        print(json.dumps(blended[0]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RICL reference utility for the ICL discussion")
    parser.add_argument("--workload-script", type=Path, required=True)
    parser.add_argument("--demo-bank", type=Path, default=None)
    parser.add_argument("--mode", choices=("paper-report", "reference-demo"), default="paper-report")
    parser.add_argument("--num-neighbors", type=int, default=4)
    parser.add_argument("--lambda-weight", type=float, default=10.0)
    args = parser.parse_args(argv)

    if not args.workload_script.exists():
        raise SystemExit(f"missing workload script: {args.workload_script}")
    task_sequence = load_octo_task_sequence(args.workload_script)
    if len(task_sequence) < 5:
        raise SystemExit("expected at least five tasks in the single-arm workload")

    first_task = task_sequence[0]
    fifth_task = task_sequence[4]

    if args.mode == "paper-report":
        print_paper_report(task_sequence, first_task, fifth_task)
        return 0

    if args.demo_bank is not None:
        demos = load_demo_bank(args.demo_bank)
        if len(demos) < 2:
            raise SystemExit("demo bank must contain at least two demos")
        model = RiclReference(lambda_weight=args.lambda_weight, num_neighbors=args.num_neighbors)
        retrieved = model.retrieve(demos[0].embedding, demos[1:])
        blended = model.interpolate_action_chunks(demos[0].action_chunk, retrieved)
        print("[RICL Reference]")
        print(f"loaded_demo_bank={args.demo_bank}")
        print(f"retrieved_neighbors={len(retrieved)}")
        for item in retrieved:
            print(f"- {item.demo.name}: distance={item.distance:.6f}")
        if blended:
            print(json.dumps(blended[0]))
        return 0

    run_reference_demo(task_sequence, args.num_neighbors, args.lambda_weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
