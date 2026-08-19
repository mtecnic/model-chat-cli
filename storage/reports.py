"""Markdown report renderers for all result types.

Pure functions: given the engine's data structures (or pre-normalized dicts),
produce GitHub-friendly markdown. The UI displays the same text; publishing
reuses the same files.
"""
import datetime
from typing import Any, Dict, List, Optional


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a markdown table from headers + rows (cells pre-escaped)."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _esc(text: str) -> str:
    """Escape pipe characters for markdown table cells."""
    return str(text).replace("|", "\\|")


def _fmt_wins(v: float) -> str:
    return f"{v:.1f}" if v % 1 else str(int(v))


# ── Chat ─────────────────────────────────────────────────────────────────────


def chat_markdown(model: str, server: str, system_prompt: str,
                  messages: List[dict]) -> str:
    """Render a chat transcript as markdown."""
    lines: List[str] = []
    lines.append(f"# Chat with {model}")
    lines.append("")
    lines.append(f"**Server:** {server}  ")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    if system_prompt:
        lines.append("**System Prompt:**")
        lines.append("")
        lines.append("> " + system_prompt.replace("\n", "\n> "))
        lines.append("")
    lines.append("---")
    lines.append("")
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"## {role}")
        lines.append("")
        if role == "ASSISTANT" and msg.get("thinking"):
            lines.append("<details>")
            lines.append("<summary>Thinking</summary>")
            lines.append("")
            lines.append(msg["thinking"])
            lines.append("")
            lines.append("</details>")
            lines.append("")
        lines.append(msg.get("content", ""))
        lines.append("")
    return "\n".join(lines)


# ── Stress ───────────────────────────────────────────────────────────────────


def stress_markdown(mode: str, model: str, server: str, stats: Any,
                    results: List[Any], extra: Optional[dict] = None) -> str:
    """Render a stress test report. Handles the tool-bench variant too."""
    if getattr(stats, "tasks_total", 0) > 0:
        return _tool_bench_markdown(mode, model, server, stats, results, extra)

    lines: List[str] = []
    lines.append(f"# Stress Test — {mode}")
    lines.append("")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Server:** {server}  ")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if extra:
        lines.append("")
        for key, value in extra.items():
            lines.append(f"**{key}:** {value}")
    if getattr(stats, "notes", ""):
        lines.append("")
        lines.append("> " + stats.notes.replace("\n", "\n> "))
    lines.append("")

    # Summary
    success_rate = (stats.success / stats.total * 100) if stats.total else 0
    lines.append("## Summary")
    lines.append("")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Total Requests", str(stats.total)],
            ["Successful", str(stats.success)],
            ["Failed", str(stats.failed)],
            ["Success Rate", f"{success_rate:.1f}%"],
        ],
    ))
    lines.append("")

    # Performance
    perf_rows = []
    if stats.avg_decode_tps > 0:
        perf_rows.append(["Avg Decode TPS", f"{stats.avg_decode_tps:.1f}"])
    if stats.total_throughput_tps > 0:
        label = "System Throughput (est.)" if stats.throughput_estimated else "System Throughput"
        perf_rows.append([label, f"{stats.total_throughput_tps:.1f} t/s"])
    if stats.avg_ttft > 0:
        perf_rows.append(["Avg TTFT", f"{stats.avg_ttft:.2f}s"])
    perf_rows.append(["Avg Response Time", f"{stats.avg_response_time:.2f}s"])
    if stats.total_output_tokens > 0:
        perf_rows.append(["Total Output Tokens", str(stats.total_output_tokens)])
    if stats.wall_clock_time > 0:
        perf_rows.append(["Wall Clock", f"{stats.wall_clock_time:.1f}s"])
    if stats.max_concurrent > 0:
        perf_rows.append(["Peak Concurrency", str(stats.max_concurrent)])
    if perf_rows:
        lines.append("## Performance")
        lines.append("")
        lines.append(_md_table(["Metric", "Value"], perf_rows))
        lines.append("")

    # Per-request breakdown (when prompt sizes vary)
    successful = [r for r in results if r.status == "success"]
    sizes = {r.prompt_tokens for r in successful if r.prompt_tokens > 0}
    if len(sizes) >= 2:
        lines.append("## Per-Request Breakdown")
        lines.append("")
        rows = []
        for r in successful:
            prompt_tok = str(r.prompt_tokens) if r.prompt_tokens else "—"
            out_tok = str(getattr(r, "completion_tokens", 0) or r.token_count)
            ttft = f"{r.ttft:.2f}s" if r.ttft > 0 else "—"
            tps = f"{r.decode_tps:.1f}" if r.decode_tps > 0 else "—"
            rows.append([str(r.request_id), prompt_tok, out_tok, ttft, tps, f"{r.duration:.2f}s"])
        lines.append(_md_table(
            ["#", "Prompt tok", "Output tok", "TTFT", "Decode TPS", "Duration"], rows
        ))
        lines.append("")

    # Latency distribution
    if stats.p50_response_time > 0 or stats.p50_ttft > 0 or stats.p50_decode_tps > 0:
        rows = []
        if stats.p50_response_time > 0:
            rows.append(["Response Time", f"{stats.p50_response_time:.2f}s",
                         f"{stats.p95_response_time:.2f}s" if stats.p95_response_time > 0 else "—",
                         f"{stats.p99_response_time:.2f}s" if stats.p99_response_time > 0 else "—"])
        if stats.p50_ttft > 0:
            rows.append(["TTFT", f"{stats.p50_ttft:.3f}s",
                         f"{stats.p95_ttft:.3f}s" if stats.p95_ttft > 0 else "—", "—"])
        if stats.p50_decode_tps > 0:
            rows.append(["Decode TPS", f"{stats.p50_decode_tps:.1f}", "—", "—"])
        lines.append("## Latency")
        lines.append("")
        lines.append(_md_table(["Metric", "p50", "p95", "p99"], rows))
        lines.append("")

    # Variance
    if stats.stddev_decode_tps > 0 or stats.stddev_ttft > 0:
        def cv(mean: float, sd: float) -> str:
            return f"{(sd / mean * 100):.2f}%" if mean > 0 else "—"
        rows = []
        if stats.avg_decode_tps > 0:
            rows.append(["Decode TPS", f"{stats.avg_decode_tps:.1f}",
                         f"{stats.stddev_decode_tps:.2f}",
                         f"{stats.min_decode_tps:.1f}", f"{stats.max_decode_tps:.1f}",
                         cv(stats.avg_decode_tps, stats.stddev_decode_tps)])
        if stats.avg_ttft > 0:
            rows.append(["TTFT", f"{stats.avg_ttft:.3f}s",
                         f"{stats.stddev_ttft:.3f}s",
                         f"{stats.min_ttft:.3f}s", f"{stats.max_ttft:.3f}s",
                         cv(stats.avg_ttft, stats.stddev_ttft)])
        if stats.avg_response_time > 0 and stats.stddev_response_time > 0:
            rows.append(["Response", f"{stats.avg_response_time:.2f}s",
                         f"{stats.stddev_response_time:.2f}s", "—", "—",
                         cv(stats.avg_response_time, stats.stddev_response_time)])
        lines.append("## Variance (hardware noise)")
        lines.append("")
        lines.append(_md_table(
            ["Metric", "Mean", "Stddev", "Min", "Max", "CV%"], rows
        ))
        lines.append("")

    # Drift
    if stats.first_half_decode_tps > 0 and stats.second_half_decode_tps > 0:
        def delta(first: float, second: float) -> str:
            if first <= 0:
                return "—"
            pct = (second - first) / first * 100
            return f"{'+' if pct >= 0 else ''}{pct:.2f}%"
        rows = [["Decode TPS", f"{stats.first_half_decode_tps:.1f}",
                 f"{stats.second_half_decode_tps:.1f}",
                 delta(stats.first_half_decode_tps, stats.second_half_decode_tps)]]
        if stats.first_half_ttft > 0 and stats.second_half_ttft > 0:
            rows.append(["TTFT", f"{stats.first_half_ttft:.3f}s",
                         f"{stats.second_half_ttft:.3f}s",
                         delta(stats.first_half_ttft, stats.second_half_ttft)])
        lines.append("## Drift (first half vs second half)")
        lines.append("")
        lines.append(_md_table(["Metric", "First half", "Second half", "Delta"], rows))
        lines.append("")

    # Errors
    if stats.errors:
        lines.append(f"## Errors ({len(stats.errors)})")
        lines.append("")
        for e in stats.errors[:20]:
            lines.append(f"- {e}")
        if len(stats.errors) > 20:
            lines.append(f"- … and {len(stats.errors) - 20} more")
        lines.append("")

    return "\n".join(lines)


def _tool_bench_markdown(mode: str, model: str, server: str, stats: Any,
                         results: List[Any], extra: Optional[dict] = None) -> str:
    """Render the tool-bench specific report."""
    lines: List[str] = []
    lines.append(f"# Tool Calling Benchmark — {model}")
    lines.append("")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Server:** {server}  ")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if extra:
        lines.append("")
        for key, value in extra.items():
            lines.append(f"**{key}:** {value}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Tasks", str(stats.tasks_total)],
            ["Passed", str(stats.tasks_passed)],
            ["Pass Rate", f"{stats.pass_rate * 100:.1f}%"],
            ["Avg Iterations / task", f"{stats.avg_agent_iterations:.1f}"],
            ["Avg Tool Calls / task", f"{stats.avg_tool_calls_per_task:.1f}"],
            ["Avg Wall Time / task", f"{stats.avg_response_time:.2f}s"],
            ["Total Wall Clock", f"{stats.wall_clock_time:.1f}s"],
        ],
    ))
    lines.append("")

    by_diff: Dict[str, List[Any]] = {"easy": [], "medium": [], "hard": []}
    for r in results:
        d = getattr(r, "task_difficulty", "")
        if d in by_diff:
            by_diff[d].append(r)
    if any(by_diff.values()):
        rows = []
        for d, rs in by_diff.items():
            if not rs:
                continue
            passed = sum(1 for r in rs if r.task_passed)
            rate = passed / len(rs) * 100
            rows.append([d.capitalize(), str(passed), str(len(rs)), f"{rate:.0f}%"])
        lines.append("## By Difficulty")
        lines.append("")
        lines.append(_md_table(["Difficulty", "Passed", "Total", "Rate"], rows))
        lines.append("")

    total_malformed = sum(getattr(r, "malformed_calls", 0) for r in results)
    total_unknown = sum(getattr(r, "unknown_tool_calls", 0) for r in results)
    total_empty = sum(getattr(r, "empty_responses", 0) for r in results)
    if total_malformed or total_unknown or total_empty:
        rows = []
        if total_malformed:
            rows.append(["Malformed args", str(total_malformed),
                         "Tool call had un-parseable JSON args"])
        if total_unknown:
            rows.append(["Unknown tool name", str(total_unknown),
                         "Tool name didn't match catalog"])
        if total_empty:
            rows.append(["Empty responses", str(total_empty),
                         "Model returned neither content nor tool_calls"])
        lines.append("## Model Diagnostics")
        lines.append("")
        lines.append(_md_table(["Issue", "Count", "What it means"], rows))
        lines.append("")

    lines.append("## Per-Task Results")
    lines.append("")
    rows = []

    def sort_key(r):
        return (0 if not r.task_passed else 1, r.task_id)

    for r in sorted(results, key=sort_key):
        if not r.task_id:
            continue
        budget = " (over)" if r.exceeded_budget else ""
        rows.append([
            _esc(r.task_id),
            (getattr(r, "task_difficulty", "") or "")[:1].upper(),
            "PASS" if r.task_passed else "FAIL",
            f"{r.agent_iterations}{budget}",
            str(r.tool_calls_made),
            "yes" if r.called_expected else "no",
            "no-forbidden" if not r.called_forbidden else "FORBIDDEN",
            "yes" if r.answer_check_ok else "no",
            f"{r.duration:.1f}s",
            _esc(getattr(r, "failure_reason", "") or ""),
        ])
    lines.append(_md_table(
        ["Task", "Diff", "Pass", "Iter", "Calls", "Tool", "Forbilled", "Ans", "Time", "Reason"],
        rows,
    ))
    lines.append("")

    if stats.errors:
        lines.append(f"## Errors ({len(stats.errors)})")
        lines.append("")
        for e in stats.errors[:20]:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)


# ── Prompt Arena ─────────────────────────────────────────────────────────────


def prompt_arena_markdown(model: str, server: str, data: dict) -> str:
    """Render a prompt arena result (tournament or multi-round).

    data contract:
      mode: "tournament" | "multi_round"
      prompts: {key: name}
      tournament: question, responses, matchups, rankings, avg_scores, winner
      multi_round: questions, rounds (list of {question, rankings, winner, avg_scores}),
                   totals, win_rates, avg_scores
    """
    prompts = data.get("prompts", {})
    name_of = lambda k: prompts.get(k, k)
    lines: List[str] = []
    lines.append("# Prompt Arena")
    lines.append("")
    lines.append(f"**Model (generator/judge):** {model}  ")
    lines.append(f"**Server:** {server}  ")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    mode = data.get("mode", "tournament")
    if mode == "multi_round":
        totals = data.get("totals", {})
        win_rates = data.get("win_rates", {})
        avg_scores = data.get("avg_scores", {})
        order = sorted(totals.items(), key=lambda x: (x[1], avg_scores.get(x[0], 0)), reverse=True)
        lines.append(f"## Final Standings ({len(data.get('questions', []))} rounds)")
        lines.append("")
        rows = []
        for rank, (key, wins) in enumerate(order, 1):
            wr = win_rates.get(key, 0)
            avg = avg_scores.get(key, 0)
            rows.append([f"#{rank}", name_of(key), _fmt_wins(wins),
                         f"{wr * 100:.1f}%", f"{avg:.1f}" if avg > 0 else "—"])
        lines.append(_md_table(["Rank", "Prompt", "Total Wins", "Win Rate", "Avg Score"], rows))
        lines.append("")

        rounds = data.get("rounds", [])
        if rounds:
            lines.append("## Round Winners")
            lines.append("")
            rows = []
            for i, rnd in enumerate(rounds, 1):
                w = rnd.get("winner", "")
                rows.append([f"R{i}", _esc((rnd.get("question") or "")[:60]),
                             name_of(w) if w else "—"])
            lines.append(_md_table(["Round", "Question", "Winner"], rows))
            lines.append("")
        return "\n".join(lines)

    # ── single tournament ──
    q = data.get("question", "")
    lines.append("## Question")
    lines.append("")
    lines.append(q)
    lines.append("")

    rankings = data.get("rankings", {})
    avg_scores = data.get("avg_scores", {})
    order = sorted(rankings.items(), key=lambda x: (x[1], avg_scores.get(x[0], 0)), reverse=True)
    if data.get("winner"):
        lines.append(f"**Winner:** {name_of(data['winner'])}")
        lines.append("")

    lines.append("## Final Standings")
    lines.append("")
    rows = []
    for rank, (key, wins) in enumerate(order, 1):
        avg = avg_scores.get(key, 0)
        resp = next((r for r in data.get("responses", []) if r.get("prompt_key") == key), None)
        t = f"{resp['duration']:.1f}s" if resp and resp.get("duration") else "—"
        rows.append([f"#{rank}", name_of(key), _fmt_wins(wins),
                     f"{avg:.1f}" if avg > 0 else "—", t])
    lines.append(_md_table(["Rank", "Prompt", "Wins", "Avg Score", "Time"], rows))
    lines.append("")

    responses = data.get("responses", [])
    if responses:
        lines.append("## Responses")
        lines.append("")
        for r in responses:
            status = r.get("status", "")
            lines.append(f"### {name_of(r.get('prompt_key', ''))} "
                         f"({status})")
            lines.append("")
            lines.append(r.get("response", "") or "*(empty)*")
            lines.append("")

    matchups = data.get("matchups", [])
    if matchups:
        lines.append("## Matchups")
        lines.append("")
        rows = []
        for m in matchups:
            w = m.get("winner", "TIE")
            verdict = f"{m.get('a', '')} > {m.get('b', '')}" if w == "A" else (
                f"{m.get('a', '')} < {m.get('b', '')}" if w == "B"
                else f"{m.get('a', '')} = {m.get('b', '')}")
            rows.append([_esc(verdict),
                         f"{m.get('score_a', 0)}–{m.get('score_b', 0)}",
                         _esc((m.get('explanation') or "")[:160])])
        lines.append(_md_table(["Matchup", "Score", "Judge Explanation"], rows))
        lines.append("")

    return "\n".join(lines)


# ── Model Arena ──────────────────────────────────────────────────────────────


def model_arena_markdown(models: List[str], data: dict) -> str:
    """Render a multi-model arena result.

    data contract:
      mode: str
      judge: str (label or "")
      blind: bool
      criteria: [str]
      rounds: [{prompt, responses: [{label, text, token_count, total_time, tps, ttft, error}],
                winner_label, judge_explanation}]
      final_scores: {label: wins}
    """
    lines: List[str] = []
    lines.append("# Model Arena")
    lines.append("")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    for i, m in enumerate(models, 1):
        lines.append(f"{i}. {m}")
    lines.append("")
    meta: List[str] = []
    if data.get("mode"):
        meta.append(f"Mode: {data['mode']}")
    if data.get("judge"):
        meta.append(f"Judge: {data['judge']}")
    if data.get("blind"):
        meta.append("Blind: yes")
    if meta:
        lines.append("  ".join(meta) + "  ")
        lines.append("")
    criteria = data.get("criteria") or []
    if criteria:
        lines.append("## Criteria")
        lines.append("")
        for c in criteria:
            lines.append(f"- {c}")
        lines.append("")

    scores = data.get("final_scores", {})
    if scores:
        lines.append("## Scores")
        lines.append("")
        rows = [[label, str(int(score)) if score == int(score) else f"{score:.1f}"]
                for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        lines.append(_md_table(["Model", "Wins"], rows))
        lines.append("")

    rounds = data.get("rounds", [])
    for rr in rounds:
        lines.append(f"## Round {rr.get('round_num', '')}")
        lines.append("")
        lines.append(f"**Prompt:** {rr.get('prompt', '')}")
        if rr.get("winner_label"):
            lines.append("")
            lines.append(f"**Winner:** {rr['winner_label']}")
        if rr.get("judge_explanation"):
            lines.append("")
            lines.append(f"**Judge:** {rr['judge_explanation']}")
        lines.append("")
        for resp in rr.get("responses", []):
            lines.append(f"### {resp.get('label', '?')}")
            lines.append("")
            lines.append(resp.get("text", "") or "*(empty)*")
            lines.append("")
            lines.append(f"*{resp.get('token_count', 0)} tok, "
                         f"{resp.get('total_time', 0):.1f}s, "
                         f"{resp.get('tps', 0):.1f} t/s, "
                         f"TTFT {resp.get('ttft', 0):.2f}s*")
            lines.append("")

    return "\n".join(lines)
