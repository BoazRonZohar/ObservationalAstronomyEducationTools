#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCO_Submit_Request.py — validate and submit an LCO observation request
through the API.

Where the request file comes from: open one of the planning tools in
Observation_Planning/ (the LCO ones), pick a target, and press "Download JSON".
Put the file it produces in this folder. No request files ship with this
toolkit — a request carries a proposal code, target coordinates and observing
windows, all of which are yours and not anyone else's.

The simple way — in Spyder, just press F5. Nothing needs editing:
    the script lists every request file in the folder, asks you to pick a
    number, asks whether to validate only or also submit, and then asks for
    your token.

From the command line, if you prefer:
    python LCO_Submit_Request.py my_request.json            validate only
    python LCO_Submit_Request.py my_request.json --submit   validate, then submit

Take your token from https://observe.lco.global/accounts/profile/

The script always runs the validate endpoint first. That endpoint submits
nothing; it returns the duration of the request and any errors. An actual
submission happens only with --submit, and only after you confirm it
explicitly at the keyboard.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import argparse
import getpass
import glob
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    sys.exit("The requests library is missing.  Install it with:  "
             "pip install requests")

API = "https://observe.lco.global/api"
PORTAL = "https://observe.lco.global"


# ------------------------------------------------------- interactive choice

def discover(folder):
    """Find every JSON file in the folder that looks like an observation
    request."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            r = d["requests"][0]
            cfg = r["configurations"][0]
            ic = cfg["instrument_configs"][0]
        except Exception:
            continue
        out.append({
            "path": p, "file": os.path.basename(p),
            "target": cfg["target"]["name"],
            "n": int(ic["exposure_count"]), "exp": float(ic["exposure_time"]),
            "filt": ic.get("optical_elements", {}).get("filter", "?"),
            "wins": len(r.get("windows", [])),
            "first": (r.get("windows") or [{}])[0].get("start", "")[:16],
        })
    return out


def choose_payload(folder):
    items = discover(folder)
    if not items:
        sys.exit("No request files found in the folder:\n  " + folder +
                 "\n\nTo make one: open a planning tool from "
                 "Observation_Planning/ in your browser,\n"
                 "pick a target, fill in your proposal code, press "
                 "\"Download JSON\",\nand put the file here.")
    print("Request files found:\n")
    for i, it in enumerate(items, 1):
        print("  %2d. %-26s target %-12s  %4dx%-4.0fs  filter %-4s  "
              "%2d windows  from %s"
              % (i, it["file"], it["target"], it["n"], it["exp"],
                 it["filt"], it["wins"], it["first"]))
    while True:
        try:
            a = input("\nPick a number (Enter to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit("Cancelled.")
        if not a:
            sys.exit("Cancelled.")
        if a.isdigit() and 1 <= int(a) <= len(items):
            it = items[int(a) - 1]
            print("Selected: %s  (target %s)" % (it["file"], it["target"]))
            return it["path"]
        print("Not a valid number, try again.")


def clean_token(t):
    """Strip spaces, quotes, and the word Token if it was pasted along with
    the value."""
    t = (t or "").strip().strip('"').strip("'").strip()
    if t.lower().startswith("token "):
        t = t[6:].strip()
    return t


def token_problem(t):
    """Return a description of what is wrong, or None if the token looks
    valid."""
    if not t:
        return "empty"
    try:
        t.encode("ascii")
    except UnicodeEncodeError:
        bad = [ch for ch in t if ord(ch) > 127][:4]
        return ("contains non-Latin characters (" + " ".join(bad) +
                ") — most likely some surrounding text was copied, not the "
                "token itself")
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", t):
        return ("contains unexpected characters (a space, punctuation) — most "
                "likely some surrounding text was copied along with it")
    if len(t) < 20:
        return ("too short (%d characters). An observatory token is normally "
                "40 characters" % len(t))
    return None


def ask_token():
    """Ask for the token, validating it and retrying."""
    print("No valid token found in the environment.")
    print("Take yours from %s/accounts/profile/" % PORTAL)
    print("(pasting into the Spyder console sometimes fails — if it does, "
          "see the note at the end)")
    for attempt in range(3):
        try:
            try:
                raw = getpass.getpass("\nPaste the token: ")
            except Exception:
                raw = input("\nPaste the token: ")
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit("Cancelled.")
        t = clean_token(raw)
        why = token_problem(t)
        if not why:
            return t
        print("Not valid — " + why)
    sys.exit("Three attempts failed.\n"
             "A way that always works in Spyder — type this in the console, "
             "once:\n"
             "    import os; os.environ['LCO_TOKEN'] = 'your-token-here'\n"
             "then run again with F5.")


def choose_mode():
    print("\nWhat would you like to do?")
    print("  1 = validate only — no observation is submitted")
    print("  2 = validate, then submit — you will be asked to confirm first")
    while True:
        try:
            a = (input("Choose [1]: ").strip() or "1")
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit("Cancelled.")
        if a in ("1", "2"):
            return a == "2"
        print("Type 1 or 2.")


# ----------------------------------------------------------------- helpers

def headers(token):
    return {"Authorization": "Token " + token, "Content-Type": "application/json"}


def hhmm(seconds):
    """seconds -> '3h 34m  (214 min)'"""
    s = int(round(seconds))
    return "%dh %02dm  (%d min)" % (s // 3600, (s % 3600) // 60, s // 60)


def post(url, payload, token):
    try:
        r = requests.post(url, json=payload, headers=headers(token), timeout=60)
    except requests.RequestException as exc:
        sys.exit("Network error: %s" % exc)
    if r.status_code == 401:
        sys.exit("The token was rejected (401). Check LCO_TOKEN against the "
                 "profile page in the portal.")
    if r.status_code == 429:
        sys.exit("You have hit the API rate limit (429). Wait a minute and "
                 "try again.")
    try:
        return r.status_code, r.json()
    except ValueError:
        sys.exit("The response was not JSON (code %s):\n%s"
                 % (r.status_code, r.text[:1000]))


def walk_errors(node, path=""):
    """Error messages come back nested — flatten them into readable lines."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += walk_errors(v, "%s/%s" % (path, k) if path else str(k))
    elif isinstance(node, list):
        if node and all(isinstance(x, str) for x in node):
            out.append((path, "; ".join(node)))
        else:
            for i, v in enumerate(node):
                out += walk_errors(v, "%s[%d]" % (path, i))
    elif isinstance(node, str):
        out.append((path, node))
    return out


def describe(payload):
    """A short summary of what is about to be sent, so you can see it is what
    you meant."""
    print("=" * 66)
    print("Request name :", payload.get("name"))
    print("Proposal     :", payload.get("proposal"))
    print("IPP          :", payload.get("ipp_value"))
    for ri, req in enumerate(payload.get("requests", []), 1):
        wins = req.get("windows", [])
        print("\nRequest %d — %d windows" % (ri, len(wins)))
        for cfg in req.get("configurations", []):
            tgt = cfg.get("target", {})
            con = cfg.get("constraints", {})
            print("  target      :", tgt.get("name"),
                  " ra=%s dec=%s" % (tgt.get("ra"), tgt.get("dec")))
            print("  instrument  :", cfg.get("instrument_type"))
            for ic in cfg.get("instrument_configs", []):
                n = int(ic.get("exposure_count", 0))
                t = float(ic.get("exposure_time", 0))
                print("  exposures   : %d x %g s   mode=%s   filter=%s   "
                      "defocus=%s"
                      % (n, t, ic.get("mode"),
                         ic.get("optical_elements", {}).get("filter"),
                         ic.get("extra_params", {}).get("defocus", "—")))
            print("  constraints : airmass<=%s  lunar_sep>=%s  lunar_phase<=%s"
                  % (con.get("max_airmass"), con.get("min_lunar_distance"),
                     con.get("max_lunar_phase")))
        for i, w in enumerate(wins, 1):
            print("   %2d. %s  →  %s" % (i, w.get("start"), w.get("end")))
    print("=" * 66)


def window_minutes(w):
    """Window length in minutes, for the format 'YYYY-MM-DD HH:MM:SS'."""
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        a = datetime.strptime(w["start"], fmt)
        b = datetime.strptime(w["end"], fmt)
    except (KeyError, ValueError):
        return None
    return (b - a).total_seconds() / 60.0


# ----------------------------------------------------------------- main

def main():
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser(
            description="Validate and submit an LCO request")
        ap.add_argument("payload", help="the request JSON file")
        ap.add_argument("--submit", action="store_true",
                        help="actually submit after a successful validation "
                             "(with keyboard confirmation)")
        ap.add_argument("--yes", action="store_true",
                        help="skip the confirmation question (for scripts)")
        args = ap.parse_args()
    else:
        # Run with no arguments (F5 in Spyder) — ask for everything at the
        # keyboard. Look for files beside the script itself, not in Spyder's
        # working directory.
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        class A:
            pass
        args = A()
        args.payload = choose_payload(here or os.getcwd())
        args.submit = choose_mode()
        args.yes = False
        print()

    # The token: from the environment first, and if it is not there, ask at
    # the keyboard. getpass does not echo the characters and does not store
    # them anywhere.
    token = clean_token(os.environ.get("LCO_TOKEN", ""))
    if token and token_problem(token):
        print("The token in the LCO_TOKEN environment variable is not valid: "
              + token_problem(token))
        token = ""
    if not token:
        token = ask_token()

    with open(args.payload, encoding="utf-8") as fh:
        payload = json.load(fh)

    describe(payload)

    # ---------- step 1: validate ----------
    print("\nSending for validation (no observation is submitted at this "
          "stage)...\n")
    code, res = post(API + "/requestgroups/validate/", payload, token)

    errs = walk_errors(res.get("errors", {}))
    durations = res.get("request_durations", {}) or {}

    reqs = durations.get("requests", [])
    total = durations.get("duration")

    ok = True
    for i, d in enumerate(reqs):
        dur = d.get("duration")
        if dur is None:
            continue
        wins = payload["requests"][i].get("windows", [])
        print("Request %d — observation duration: %s" % (i + 1, hhmm(dur)))
        shortest = None
        for w in wins:
            m = window_minutes(w)
            if m is not None and (shortest is None or m < shortest):
                shortest = m
        if shortest is not None:
            slack = shortest - dur / 60.0
            # The slack is how far the scheduler can move the block inside the
            # window. It is taken out of the baseline before the event, so you
            # do not want it too large — but not zero either, or the scheduler
            # has no freedom at all.
            if slack < 0:
                mark = "too short"; ok = False
            elif slack < 2:
                # The observatory rejects a request whose duration equals the
                # visible time, and accepts one already at four minutes.
                mark = "tight"; ok = False
            else:
                mark = "ok"
            print("   shortest window: %d min   slack: %+d min   [%s]"
                  % (shortest, round(slack), mark))
            if slack < 0:
                print("   >> The block is longer than the window. Reduce "
                      "exposure_count or lengthen the window.")
            elif slack < 2:
                print("   >> The observatory requires the duration to be "
                      "strictly less than the visible time."
                      " Reduce exposure_count a little.")
            elif slack > 25:
                print("   >> Large slack. Note that it comes out of the "
                      "baseline before the event —"
                      " you can raise exposure_count to shrink it.")
    if total:
        print("\nTotal time that will be charged to the allocation: %s"
              % hhmm(total))

    if errs:
        ok = False
        print("\nValidation errors:")
        for path, msg in errs:
            print("  • %s: %s" % (path, msg))
    else:
        print("\nValidation passed with no errors.")

    if not args.submit:
        print("\n(Validation only. To submit as well, run again with --submit)")
        return 0 if ok else 1

    if errs:
        sys.exit("\nNot submitting — there are validation errors to fix first.")

    if not ok and not args.yes:
        print("\nNote: one of the windows has less than a quarter hour of "
              "slack.")

    if not args.yes:
        try:
            ans = input("\nSubmit the request now? Type  yes  to confirm: "
                        ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if ans != "yes":
            print("Cancelled. Nothing was submitted.")
            return 1

    # ---------- step 2: submit ----------
    code, res = post(API + "/requestgroups/", payload, token)
    if code in (200, 201):
        rgid = res.get("id")
        print("\nSubmitted successfully.  Request group id: %s" % rgid)
        if rgid:
            print("Track it: %s/requestgroups/%s/" % (PORTAL, rgid))
        for r in res.get("requests", []):
            print("   request %s — state %s" % (r.get("id"), r.get("state")))
    else:
        print("\nThe submission failed (code %s):" % code)
        print(json.dumps(res, ensure_ascii=False, indent=2)[:2000])
        return 1
    return 0


if __name__ == "__main__":
    # In Spyder, calling sys.exit shows "An exception has occurred".
    # When run interactively, just print the message and finish quietly.
    _interactive = len(sys.argv) <= 1
    try:
        _code = main()
    except SystemExit as _exc:
        if isinstance(_exc.code, str):
            print(_exc.code)
            _code = 1
        else:
            _code = _exc.code or 0
    if not _interactive:
        sys.exit(_code)
