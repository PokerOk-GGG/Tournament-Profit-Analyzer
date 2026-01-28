#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tournament.py — PokerOK Tournament Profit Analyzer
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


APP_ROOM = "PokerOK"
DEFAULT_CURRENCY = "USD"
DEFAULT_STORE = "tournaments.json"

TOUR_TYPES = {"MTT", "Bounty", "Turbo", "Hyper", "Satellite", "Other"}


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Tournament:
    id: str
    date: str  # YYYY-MM-DD
    room: str
    type: str
    buyin: float
    rake: float
    total_cost: float
    cash: float
    currency: str = DEFAULT_CURRENCY
    field: Optional[int] = None
    place: Optional[int] = None
    notes: Optional[str] = None
    reentries: int = 0  # number of re-entries (0 means single entry)

    def profit(self) -> float:
        return self.cash - self.total_cost

    def roi(self) -> float:
        return (self.profit() / self.total_cost) if self.total_cost > 0 else 0.0


# ----------------------------
# Utilities: parsing/validation
# ----------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_date_yyyy_mm_dd(s: str) -> str:
    try:
        dt.date.fromisoformat(s)
    except Exception:
        raise ValueError("Invalid date format, expected YYYY-MM-DD")
    return s


def parse_nonneg_float(x: Any, field_name: str) -> float:
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"{field_name} must be a number")
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return v


def parse_nonneg_int(x: Any, field_name: str) -> int:
    try:
        v = int(x)
    except Exception:
        raise ValueError(f"{field_name} must be an integer")
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return v


def parse_optional_pos_int(x: Any, field_name: str) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        v = int(x)
    except Exception:
        raise ValueError(f"{field_name} must be an integer")
    if v <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return v


def normalize_type(t: str) -> str:
    if not t:
        return "Other"
    # allow case-insensitive matching
    t_clean = t.strip()
    for allowed in TOUR_TYPES:
        if allowed.lower() == t_clean.lower():
            return allowed
    return "Other"


def compute_total_cost(buyin: float, rake: float, reentries: int) -> float:
    # 1 re-entry => 2 total entries
    entries = 1 + max(0, reentries)
    return (buyin + rake) * entries


def safe_float_fmt(x: float, ndigits: int = 2) -> str:
    # avoid "-0.00"
    if abs(x) < 0.0005:
        x = 0.0
    return f"{x:.{ndigits}f}"


def safe_pct_fmt(x: float, ndigits: int = 2) -> str:
    # x is fraction (0.1 => 10%)
    return f"{x * 100:.{ndigits}f}%"


# ----------------------------
# Storage
# ----------------------------

def store_path(path: Optional[str]) -> Path:
    p = Path(path or DEFAULT_STORE)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def load_tournaments(path: Path) -> List[Tournament]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Storage format invalid: expected a list")
        out: List[Tournament] = []
        for obj in data:
            out.append(Tournament(**obj))
        return out
    except Exception as e:
        raise RuntimeError(f"Failed to load {path.name}: {e}") from e


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_tournaments(path: Path, items: List[Tournament]) -> None:
    payload = [asdict(t) for t in items]
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class Stats:
    n: int
    total_cost: float
    total_cash: float
    profit: float
    roi: float  # fraction
    itm: float  # fraction
    abi: float
    avg_profit: float
    best_profit: float
    worst_profit: float
    max_drawdown: float


def calc_max_drawdown(items: List[Tournament]) -> float:
    """
    Max drawdown over cumulative profit curve, ordered by date then id.
    Returns positive number = maximum peak-to-trough drop in profit.
    """
    if not items:
        return 0.0
    # sort by date then id for stable order
    def key(t: Tournament) -> Tuple[str, str]:
        return (t.date, t.id)

    curve = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(items, key=key):
        curve += t.profit()
        peak = max(peak, curve)
        dd = peak - curve
        max_dd = max(max_dd, dd)
    return max_dd


def compute_stats(items: List[Tournament]) -> Stats:
    n = len(items)
    if n == 0:
        return Stats(
            n=0,
            total_cost=0.0,
            total_cash=0.0,
            profit=0.0,
            roi=0.0,
            itm=0.0,
            abi=0.0,
            avg_profit=0.0,
            best_profit=0.0,
            worst_profit=0.0,
            max_drawdown=0.0,
        )

    total_cost = sum(t.total_cost for t in items)
    total_cash = sum(t.cash for t in items)
    profit = total_cash - total_cost
    roi = (profit / total_cost) if total_cost > 0 else 0.0
    itm = (sum(1 for t in items if t.cash > 0) / n) if n > 0 else 0.0
    abi = (total_cost / n) if n > 0 else 0.0
    avg_profit = (profit / n) if n > 0 else 0.0
    profits = [t.profit() for t in items]
    best_profit = max(profits) if profits else 0.0
    worst_profit = min(profits) if profits else 0.0
    max_dd = calc_max_drawdown(items)

    return Stats(
        n=n,
        total_cost=total_cost,
        total_cash=total_cash,
        profit=profit,
        roi=roi,
        itm=itm,
        abi=abi,
        avg_profit=avg_profit,
        best_profit=best_profit,
        worst_profit=worst_profit,
        max_drawdown=max_dd,
    )


def format_stats_block(title: str, s: Stats, currency: str = DEFAULT_CURRENCY) -> str:
    lines = []
    lines.append(f"{title}")
    lines.append("-" * max(16, len(title)))
    lines.append(f"Tournaments : {s.n}")
    lines.append(f"Total Cost  : {safe_float_fmt(s.total_cost)} {currency}")
    lines.append(f"Total Cash  : {safe_float_fmt(s.total_cash)} {currency}")
    lines.append(f"Profit      : {safe_float_fmt(s.profit)} {currency}")
    lines.append(f"ROI         : {safe_pct_fmt(s.roi)}")
    lines.append(f"ITM         : {safe_pct_fmt(s.itm)}")
    lines.append(f"ABI         : {safe_float_fmt(s.abi)} {currency}")
    lines.append(f"Avg Profit  : {safe_float_fmt(s.avg_profit)} {currency}/tourney")
    lines.append(f"Best Result : {safe_float_fmt(s.best_profit)} {currency}")
    lines.append(f"Worst Result: {safe_float_fmt(s.worst_profit)} {currency}")
    lines.append(f"Max DD      : {safe_float_fmt(s.max_drawdown)} {currency}")
    return "\n".join(lines)


def top_k(items: List[Tournament], k: int, reverse: bool = True) -> List[Tournament]:
    return sorted(items, key=lambda t: t.profit(), reverse=reverse)[:k]


def print_top(items: List[Tournament], k: int = 5) -> None:
    if not items:
        print("No tournaments in selection.")
        return

    best = top_k(items, k=k, reverse=True)
    worst = top_k(items, k=k, reverse=False)

    def row(t: Tournament) -> str:
        return (
            f"{t.date} | {t.type:<9} | cost={safe_float_fmt(t.total_cost):>8} "
            f"| cash={safe_float_fmt(t.cash):>8} | profit={safe_float_fmt(t.profit()):>9} "
            f"| roi={safe_pct_fmt(t.roi()):>8}"
        )

    print("\nTop wins:")
    for t in best:
        print("  " + row(t))

    print("\nTop losses:")
    for t in worst:
        print("  " + row(t))


# ----------------------------
# Filtering
# ----------------------------

def parse_optional_date(s: Optional[str], name: str) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return parse_date_yyyy_mm_dd(s)


def filter_items(
    items: List[Tournament],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_buyin: Optional[float] = None,
    max_buyin: Optional[float] = None,
    ttype: Optional[str] = None,
    currency: Optional[str] = None,
) -> List[Tournament]:
    out: List[Tournament] = []

    ttype_norm = normalize_type(ttype) if ttype else None
    # NOTE: buy-in filter is applied to total_cost (as per ABI logic).
    for t in items:
        if date_from and t.date < date_from:
            continue
        if date_to and t.date > date_to:
            continue
        if min_buyin is not None and t.total_cost < min_buyin:
            continue
        if max_buyin is not None and t.total_cost > max_buyin:
            continue
        if ttype_norm and t.type != ttype_norm:
            continue
        if currency and t.currency.upper() != currency.upper():
            continue
        out.append(t)
    return out


# ----------------------------
# Interactive input helpers
# ----------------------------

def prompt_str(label: str, default: Optional[str] = None) -> str:
    if default is not None:
        s = input(f"{label} [{default}]: ").strip()
        return s if s else default
    return input(f"{label}: ").strip()


def prompt_float(label: str, default: Optional[float] = None) -> float:
    while True:
        raw = prompt_str(label, None if default is None else safe_float_fmt(default))
        try:
            return parse_nonneg_float(raw, label)
        except Exception as e:
            print(f"  {e}")


def prompt_int(label: str, default: Optional[int] = None) -> int:
    while True:
        raw = prompt_str(label, None if default is None else str(default))
        try:
            return parse_nonneg_int(raw, label)
        except Exception as e:
            print(f"  {e}")


def prompt_optional_int(label: str) -> Optional[int]:
    raw = input(f"{label} (blank to skip): ").strip()
    if not raw:
        return None
    try:
        return parse_optional_pos_int(raw, label)
    except Exception as e:
        print(f"  {e}")
        return prompt_optional_int(label)


# ----------------------------
# Commands
# ----------------------------

def cmd_add(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    items = load_tournaments(path)

    try:
        date = parse_date_yyyy_mm_dd(args.date) if args.date else None
    except Exception as e:
        die(str(e))

    if not date:
        while True:
            try:
                date = parse_date_yyyy_mm_dd(prompt_str("Date (YYYY-MM-DD)", dt.date.today().isoformat()))
                break
            except Exception as e:
                print(f"  {e}")

    ttype = normalize_type(args.type) if args.type else normalize_type(prompt_str(f"Type {sorted(TOUR_TYPES)}", "MTT"))
    currency = (args.currency or DEFAULT_CURRENCY).upper()

    # buyin/rake/cash
    def get_float_arg(name: str, default: float = 0.0) -> float:
        val = getattr(args, name)
        if val is None:
            return prompt_float(name.capitalize(), default)
        try:
            return parse_nonneg_float(val, name)
        except Exception as e:
            die(str(e))

    buyin = get_float_arg("buyin", 5.0)
    rake = get_float_arg("rake", 0.5)
    cash = get_float_arg("cash", 0.0)

    reentries = args.reentries
    if reentries is None:
        reentries = prompt_int("Re-entries (0 = none)", 0)
    else:
        try:
            reentries = parse_nonneg_int(reentries, "reentries")
        except Exception as e:
            die(str(e))

    field = args.field
    if field is None:
        field = prompt_optional_int("Field size")
    else:
        try:
            field = parse_optional_pos_int(field, "field")
        except Exception as e:
            die(str(e))

    place = args.place
    if place is None:
        place = prompt_optional_int("Place")
    else:
        try:
            place = parse_optional_pos_int(place, "place")
        except Exception as e:
            die(str(e))

    notes = args.notes
    if notes is None:
        notes = input("Notes (blank to skip): ").strip() or None

    total_cost = compute_total_cost(buyin, rake, reentries)

    t = Tournament(
        id=str(uuid.uuid4()),
        date=date,
        room=APP_ROOM,
        type=ttype,
        buyin=buyin,
        rake=rake,
        total_cost=total_cost,
        cash=cash,
        currency=currency,
        field=field,
        place=place,
        notes=notes,
        reentries=reentries,
    )

    items.append(t)
    save_tournaments(path, items)

    print("Added tournament.")
    print(f"  Profit: {safe_float_fmt(t.profit())} {t.currency}")
    print(f"  ROI   : {safe_pct_fmt(t.roi())}")


def cmd_stats(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    items = load_tournaments(path)

    # If multiple currencies exist, keep default currency display but mention it.
    currency = args.currency or (items[0].currency if items else DEFAULT_CURRENCY)
    if args.currency:
        items = filter_items(items, currency=args.currency)

    s = compute_stats(items)
    print(format_stats_block("PokerOK — Overall Stats", s, currency=currency.upper()))
    if s.n:
        print_top(items, k=5)


def cmd_report(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    items = load_tournaments(path)

    try:
        dfrom = parse_optional_date(args.date_from, "--from")
        dto = parse_optional_date(args.date_to, "--to")
        min_b = float(args.min_buyin) if args.min_buyin is not None else None
        max_b = float(args.max_buyin) if args.max_buyin is not None else None
        if min_b is not None and min_b < 0:
            raise ValueError("--min-buyin must be >= 0")
        if max_b is not None and max_b < 0:
            raise ValueError("--max-buyin must be >= 0")
    except Exception as e:
        die(str(e))

    sel = filter_items(
        items,
        date_from=dfrom,
        date_to=dto,
        min_buyin=min_b,
        max_buyin=max_b,
        ttype=args.type,
        currency=args.currency,
    )

    currency = args.currency or (sel[0].currency if sel else DEFAULT_CURRENCY)
    title_parts = ["PokerOK — Report"]
    if dfrom or dto:
        title_parts.append(f"dates={dfrom or '...'}..{dto or '...'}")
    if min_b is not None or max_b is not None:
        title_parts.append(f"cost={min_b if min_b is not None else '...'}..{max_b if max_b is not None else '...'}")
    if args.type:
        title_parts.append(f"type={normalize_type(args.type)}")
    if args.currency:
        title_parts.append(f"currency={args.currency.upper()}")

    s = compute_stats(sel)
    print(format_stats_block(" | ".join(title_parts), s, currency=currency.upper()))
    if s.n:
        print_top(sel, k=5)


def normal_sample(mu: float, sigma: float) -> float:
    # Use built-in Gaussian sampler.
    return random.gauss(mu, sigma)


def percentile(values: List[float], p: float) -> float:
    """p in [0,1]. Simple linear interpolation percentile."""
    if not values:
        return 0.0
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 1:
        return xs[-1]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def cmd_simulate(args: argparse.Namespace) -> None:
    # Inputs
    try:
        n = int(args.n)
        if n <= 0:
            raise ValueError("--n must be > 0")
        abi = float(args.abi)
        if abi <= 0:
            raise ValueError("--abi must be > 0")
        roi = float(args.roi)  # as fraction, e.g., 0.25
        sigma = float(args.sigma)
        if sigma <= 0:
            raise ValueError("--sigma must be > 0")
        trials = int(args.trials)
        if trials <= 0:
            raise ValueError("--trials must be > 0")
        loss_threshold = float(args.loss_threshold) if args.loss_threshold is not None else None
    except Exception as e:
        die(str(e))

    mu = roi * abi
    sd = sigma * abi

    totals: List[float] = []
    neg = 0
    worse_than = 0

    for _ in range(trials):
        total = 0.0
        for _ in range(n):
            total += normal_sample(mu, sd)
        totals.append(total)
        if total < 0:
            neg += 1
        if loss_threshold is not None and total <= loss_threshold:
            worse_than += 1

    p5 = percentile(totals, 0.05)
    p50 = percentile(totals, 0.50)
    p95 = percentile(totals, 0.95)

    expected_profit = sum(totals) / len(totals)
    expected_roi = expected_profit / (n * abi) if (n * abi) > 0 else 0.0

    print("PokerOK — Variance Simulation (approx.)")
    print("--------------------------------------")
    print(f"Distance (tournaments): {n}")
    print(f"ABI: {safe_float_fmt(abi)} {args.currency.upper()}")
    print(f"Assumed ROI: {safe_pct_fmt(roi)}")
    print(f"Sigma: {safe_float_fmt(sigma)} (std = sigma*ABI = {safe_float_fmt(sd)} {args.currency.upper()})")
    print(f"Trials: {trials}")
    print("")
    print(f"Expected profit: {safe_float_fmt(expected_profit)} {args.currency.upper()}")
    print(f"Expected ROI   : {safe_pct_fmt(expected_roi)}")
    print("")
    print(f"P5  result : {safe_float_fmt(p5)} {args.currency.upper()}")
    print(f"P50 result : {safe_float_fmt(p50)} {args.currency.upper()}")
    print(f"P95 result : {safe_float_fmt(p95)} {args.currency.upper()}")
    print("")
    print(f"P(result < 0): {safe_pct_fmt(neg / trials)}")
    if loss_threshold is not None:
        print(f"P(result <= {safe_float_fmt(loss_threshold)}): {safe_pct_fmt(worse_than / trials)}")
    print("")
    print("Note: This is a simplified Normal model (approx.) and does NOT model real MTT payout structures.")


def cmd_export(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    items = load_tournaments(path)

    try:
        dfrom = parse_optional_date(args.date_from, "--from")
        dto = parse_optional_date(args.date_to, "--to")
        min_b = float(args.min_buyin) if args.min_buyin is not None else None
        max_b = float(args.max_buyin) if args.max_buyin is not None else None
    except Exception as e:
        die(str(e))

    sel = filter_items(
        items,
        date_from=dfrom,
        date_to=dto,
        min_buyin=min_b,
        max_buyin=max_b,
        ttype=args.type,
        currency=args.currency,
    )

    out_path = Path(args.out or "export.csv")
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    fields = [
        "id", "date", "room", "type", "buyin", "rake", "reentries", "total_cost", "cash",
        "profit", "roi", "currency", "field", "place", "notes"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in sel:
            row = asdict(t)
            row["profit"] = t.profit()
            row["roi"] = t.roi()
            w.writerow(row)

    print(f"Exported {len(sel)} tournaments to {out_path}")


def cmd_import(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    items = load_tournaments(path)

    csv_path = Path(args.csvfile)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        die(f"CSV file not found: {csv_path}")

    errors_path = Path(args.errors or "import_errors.csv")
    if not errors_path.is_absolute():
        errors_path = Path(__file__).resolve().parent / errors_path

    imported = 0
    failed = 0
    err_rows: List[Dict[str, Any]] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        required = {"date", "buyin", "rake", "cash", "type"}
        missing = required - set((r.fieldnames or []))
        if missing:
            die(f"CSV missing required columns: {sorted(missing)}")

        for idx, row in enumerate(r, start=2):  # header = row 1
            try:
                date = parse_date_yyyy_mm_dd(row.get("date", "").strip())
                ttype = normalize_type(row.get("type", "").strip())

                buyin = parse_nonneg_float(row.get("buyin", ""), "buyin")
                rake = parse_nonneg_float(row.get("rake", ""), "rake")
                cash = parse_nonneg_float(row.get("cash", ""), "cash")

                currency = (row.get("currency") or args.currency or DEFAULT_CURRENCY).strip().upper()
                reentries = parse_nonneg_int(row.get("reentries", "0") or 0, "reentries")

                field = parse_optional_pos_int(row.get("field", ""), "field") if row.get("field") not in (None, "") else None
                place = parse_optional_pos_int(row.get("place", ""), "place") if row.get("place") not in (None, "") else None
                notes = (row.get("notes") or "").strip() or None

                total_cost = compute_total_cost(buyin, rake, reentries)

                t = Tournament(
                    id=str(uuid.uuid4()),
                    date=date,
                    room=APP_ROOM,
                    type=ttype,
                    buyin=buyin,
                    rake=rake,
                    total_cost=total_cost,
                    cash=cash,
                    currency=currency,
                    field=field,
                    place=place,
                    notes=notes,
                    reentries=reentries,
                )
                items.append(t)
                imported += 1

            except Exception as e:
                failed += 1
                err = dict(row)
                err["_row"] = idx
                err["_error"] = str(e)
                err_rows.append(err)

    save_tournaments(path, items)

    print(f"Import done. Imported: {imported}, failed: {failed}.")
    if err_rows:
        # write errors csv
        err_fields = sorted({k for er in err_rows for k in er.keys()})
        with errors_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=err_fields)
            w.writeheader()
            for er in err_rows:
                w.writerow(er)
        print(f"Errors saved to {errors_path}")


def cmd_reset(args: argparse.Namespace) -> None:
    path = store_path(args.store)
    if not path.exists():
        print("Nothing to reset (storage file not found).")
        return

    print("This will permanently delete your local tournaments database.")
    confirm = input('Type "RESET" to confirm: ').strip()
    if confirm != "RESET":
        print("Cancelled.")
        return

    atomic_write_text(path, "[]")
    print(f"Reset done: {path}")


# ----------------------------
# Menu (optional convenience)
# ----------------------------

def interactive_menu() -> None:
    print("PokerOK Tournament Analyzer")
    print("---------------------------")
    print("1) add")
    print("2) import")
    print("3) stats")
    print("4) report")
    print("5) simulate")
    print("6) export")
    print("7) reset")
    print("0) exit")

    choice = input("Choose: ").strip()
    if choice == "0":
        return

    # Re-dispatch by rebuilding argv-style command
    mapping = {
        "1": "add",
        "2": "import",
        "3": "stats",
        "4": "report",
        "5": "simulate",
        "6": "export",
        "7": "reset",
    }
    cmd = mapping.get(choice)
    if not cmd:
        print("Unknown choice.")
        return

    # Minimal: re-run main with cmd only; args will be requested interactively when needed.
    main([cmd])


# ----------------------------
# CLI setup
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tournament.py",
        description="PokerOK MTT analyzer: track tournaments, compute ROI/ITM/ABI, and simulate variance.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Path to storage JSON (default: tournaments.json).")

    sub = p.add_subparsers(dest="command")

    # add
    pa = sub.add_parser("add", help="Add a single tournament (args or interactive).")
    pa.add_argument("--date", help="Date YYYY-MM-DD")
    pa.add_argument("--type", help="Type: MTT|Bounty|Turbo|Hyper|Satellite|Other")
    pa.add_argument("--buyin", help="Buy-in amount (without rake)")
    pa.add_argument("--rake", help="Rake/fee amount")
    pa.add_argument("--cash", help="Cash/prize (0 if no ITM)")
    pa.add_argument("--currency", help="Currency (default USD)")
    pa.add_argument("--field", help="Field size (optional)")
    pa.add_argument("--place", help="Place (optional)")
    pa.add_argument("--reentries", help="Re-entries count (0 = none)")
    pa.add_argument("--notes", help="Notes (optional)")
    pa.set_defaults(func=cmd_add)

    # import
    pi = sub.add_parser("import", help="Import tournaments from CSV.")
    pi.add_argument("csvfile", help="Path to CSV file with columns: date,buyin,rake,cash,type (+optional fields).")
    pi.add_argument("--currency", help="Default currency if CSV row has none.")
    pi.add_argument("--errors", help="Path for import errors CSV (default: import_errors.csv).")
    pi.set_defaults(func=cmd_import)

    # stats
    ps = sub.add_parser("stats", help="Show overall stats (optionally filter by currency).")
    ps.add_argument("--currency", help="Filter by currency (e.g., USD).")
    ps.set_defaults(func=cmd_stats)

    # report
    pr = sub.add_parser("report", help="Show stats for a filtered subset.")
    pr.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD (inclusive)")
    pr.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD (inclusive)")
    pr.add_argument("--min-buyin", dest="min_buyin", help="Min total_cost (inclusive)")
    pr.add_argument("--max-buyin", dest="max_buyin", help="Max total_cost (inclusive)")
    pr.add_argument("--type", help="Type filter")
    pr.add_argument("--currency", help="Currency filter")
    pr.set_defaults(func=cmd_report)

    # simulate
    pm = sub.add_parser("simulate", help="Monte-Carlo variance simulation (approx.).")
    pm.add_argument("--n", required=True, help="Number of tournaments in distance (e.g., 1000)")
    pm.add_argument("--abi", required=True, help="Average buy-in (total cost per tourney)")
    pm.add_argument("--roi", required=True, help="Expected ROI as fraction (e.g., 0.25 = 25%)")
    pm.add_argument("--sigma", required=True, help="Std multiplier in ABI units (e.g., 1.6)")
    pm.add_argument("--trials", default="5000", help="Monte-Carlo trials (default 5000)")
    pm.add_argument("--loss-threshold", default=None, help="Optional threshold (e.g., -2000) for P(result <= X)")
    pm.add_argument("--currency", default=DEFAULT_CURRENCY, help="Currency label for output (default USD)")
    pm.set_defaults(func=cmd_simulate)

    # export
    pe = sub.add_parser("export", help="Export tournaments (optionally filtered) to CSV.")
    pe.add_argument("--out", help="Output CSV filename (default: export.csv)")
    pe.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD (inclusive)")
    pe.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD (inclusive)")
    pe.add_argument("--min-buyin", dest="min_buyin", help="Min total_cost (inclusive)")
    pe.add_argument("--max-buyin", dest="max_buyin", help="Max total_cost (inclusive)")
    pe.add_argument("--type", help="Type filter")
    pe.add_argument("--currency", help="Currency filter")
    pe.set_defaults(func=cmd_export)

    # reset
    prst = sub.add_parser("reset", help='Reset local storage (requires typing "RESET").')
    prst.set_defaults(func=cmd_reset)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        interactive_menu()
        return

    try:
        args.func(args)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except RuntimeError as e:
        die(str(e), code=1)


if __name__ == "__main__":
    main()