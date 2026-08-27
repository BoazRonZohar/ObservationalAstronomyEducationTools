#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCO_Check_Status.py — check the state of your requests in the LCO portal.
READ ONLY.

This script submits nothing, cancels nothing, and changes nothing. It makes
GET requests only, and summarises:
    • how much of the proposal's allocation is used (standard hours, IPP)
    • every request group: name, id, state, when it was created
    • per request: state, duration, how many windows are left and how many
      have already passed
    • the observations the scheduler actually placed — site, telescope, time,
      and what became of them

To run: press F5 in Spyder. Nothing needs editing.
The token is read from the LCO_TOKEN environment variable if it is set;
otherwise you are asked for it at the keyboard and it is not echoed.
The reliable way in Spyder:  import os; os.environ['LCO_TOKEN'] = '...'
then F5.

The output is also written to LCO_Status_Report.txt next to this script.

Your proposal code is read from the LCO_PROPOSAL environment variable if it is
set; otherwise you are asked for it at the keyboard. Nothing is stored in this
file. You will find your codes on the proposals page of the portal, and they
look like ABCXYZ2026A-001.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import getpass
import io
import os
import re
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("The requests library is missing.  Install it with:  "
             "pip install requests")

API = "https://observe.lco.global/api"
PORTAL = "https://observe.lco.global"
REPORT = "LCO_Status_Report.txt"

FMT_IN = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
          "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")


# ------------------------------------------------- print to screen and file

_buf = io.StringIO()


def say(line=""):
    print(line)
    _buf.write(line + "\n")


# ------------------------------------------------------------------- token

def clean_token(t):
    t = (t or "").strip().strip('"').strip("'").strip()
    if t.lower().startswith("token "):
        t = t[6:].strip()
    return t


def token_problem(t):
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


def ask_proposal():
    """The proposal code, from the environment or from the keyboard.

    Deliberately not hard-coded: a proposal code identifies a particular
    programme and its allocation, and does not belong in a shared script.
    """
    p = (os.environ.get("LCO_PROPOSAL", "") or "").strip().strip('"').strip("'")
    if p:
        return p
    print("Which proposal? You will find the code on the proposals page of the")
    print("portal, %s/proposals/ — it looks like ABCXYZ2026A-001." % PORTAL)
    print("To skip this question next time:  "
          "import os; os.environ['LCO_PROPOSAL'] = '...'")
    try:
        p = input("\nProposal code: ").strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("Cancelled.")
    if not p:
        sys.exit("No proposal code given.")
    return p


def ask_token():
    print("No valid token found in the environment.")
    print("Take yours from %s/accounts/profile/" % PORTAL)
    for _ in range(3):
        try:
            try:
                raw = getpass.getpass("\nPaste the token: ")
            except Exception:
                raw = input("\nPaste the token: ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Cancelled.")
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


# ------------------------------------------------------------------ helpers

def get(path, token, **params):
    url = path if path.startswith("http") else API + path
    try:
        r = requests.get(url, headers={"Authorization": "Token " + token},
                         params=params or None, timeout=60)
    except requests.RequestException as exc:
        sys.exit("Network error: %s" % exc)
    if r.status_code == 401:
        sys.exit("The token was rejected (401). Check it against the profile "
                 "page in the portal.")
    if r.status_code == 429:
        sys.exit("You have hit the API rate limit (429). Wait a minute and "
                 "try again.")
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def parse(ts):
    if not ts:
        return None
    for f in FMT_IN:
        try:
            return datetime.strptime(ts, f)
        except ValueError:
            pass
    return None


def short(ts):
    d = parse(ts)
    return d.strftime("%d/%m %H:%M") if d else str(ts)[:16]


def hhmm(seconds):
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


# ------------------------------------------------------------------ reports

def find_allocations(p):
    """Allocations appear under a field name that has changed between portal
    versions. Rather than guess, look for any list whose members look like a
    time allocation."""
    for key in ("timeallocations", "timeallocation_set", "time_allocations"):
        v = p.get(key)
        if isinstance(v, list) and v:
            return v, key
    for key, v in p.items():
        if (isinstance(v, list) and v and isinstance(v[0], dict)
                and any("allocation" in k for k in v[0])):
            return v, key
    return [], None


def report_allocation(token, proposal):
    p = get("/proposals/%s/" % proposal, token)
    if not p:
        say("Could not read the details of proposal %s." % proposal)
        say("")
        return
    say("=" * 72)
    say("Proposal %s — %s" % (proposal, p.get("title", "")))
    say("=" * 72)
    allocs, key = find_allocations(p)
    if not allocs:
        say("  No time allocations found in the API response.")
        say("  Fields returned: " + ", ".join(sorted(p.keys())))
        say("")
        return
    for a in allocs:
        inst = a.get("instrument_types") or a.get("instrument_type") or "?"
        if isinstance(inst, list):
            inst = ", ".join(inst)
        say("  Instrument %s   semester %s" % (inst, a.get("semester", "?")))
        shown = False
        for label, used, total in (
                ("std ", a.get("std_time_used"), a.get("std_allocation")),
                ("RR  ", a.get("rr_time_used"), a.get("rr_allocation")),
                ("TC  ", a.get("tc_time_used"), a.get("tc_allocation"))):
            if total in (None, 0):
                continue
            shown = True
            try:
                pct = 100.0 * float(used) / float(total)
            except (TypeError, ValueError, ZeroDivisionError):
                pct = float("nan")
            say("    %s  %8.2f / %8.2f hours   (%.0f%%)"
                % (label, float(used or 0), float(total), pct))
        # The IPP bank is reported the other way round: how much is available,
        # not how much has been used.
        ipp_av, ipp_lim = a.get("ipp_time_available"), a.get("ipp_limit")
        if ipp_lim not in (None, 0):
            shown = True
            say("    IPP   %.2f available out of a ceiling of %.2f hours"
                % (float(ipp_av or 0), float(ipp_lim)))
        if not shown:
            say("    (unrecognised fields) " + ", ".join(sorted(a.keys())))
    say("")


def observations_for(req_id, token):
    """The scheduler re-runs the plan every few minutes, and on each run it
    cancels the previous block and creates it again. Most observations are
    therefore routine CANCELED entries — noise. Return the meaningful ones
    separately from a count of the cancelled ones."""
    d = get("/observations/", token, request_id=req_id, limit=200)
    if not d:
        return [], 0
    obs = d if isinstance(d, list) else d.get("results", [])
    live = [o for o in obs if o.get("state") != "CANCELED"]
    return live, len(obs) - len(live)


def report_groups(token, proposal, limit=50):
    d = get("/requestgroups/", token, proposal=proposal, limit=limit,
            ordering="-created")
    if not d:
        sys.exit("Could not read the list of requests.")
    groups = d.get("results", [])
    now = datetime.utcnow()
    say("Found %d request groups in the proposal.  The time now is %s UTC"
        % (d.get("count", len(groups)), now.strftime("%d/%m %H:%M")))
    say("")

    # Still alive first, then expired, and completed last.
    order = {"PENDING": 0, "WINDOW_EXPIRED": 1, "CANCELED": 2, "COMPLETED": 3}
    groups.sort(key=lambda g: (order.get(g.get("state"), 9),
                               g.get("created", "")))

    for g in groups:
        say("-" * 72)
        say("%-38s  [%s]" % (g.get("name", "?"), g.get("state", "?")))
        say("  id %s   created %s   %s/requestgroups/%s/"
            % (g.get("id"), short(g.get("created")), PORTAL, g.get("id")))
        for req in g.get("requests", []) or []:
            wins = req.get("windows", []) or []
            future = [w for w in wins if (parse(w.get("end")) or now) > now]
            say("  request %s — state %s   duration %s   "
                "windows: %d of %d still ahead"
                % (req.get("id"), req.get("state"), hhmm(req.get("duration")),
                   len(future), len(wins)))
            if future:
                nxt = min(future, key=lambda w: parse(w.get("start")) or now)
                last = max(future, key=lambda w: parse(w.get("end")) or now)
                say("     next window: %s → %s"
                    % (short(nxt.get("start")), short(nxt.get("end"))))
                say("     last window ends: %s" % short(last.get("end")))
            elif wins:
                say("     every window has passed.")
            note = req.get("observation_note")
            if note:
                say("     note: %s" % note)
            obs, ncanceled = observations_for(req.get("id"), token)
            for o in obs:
                say("       %s %s  %s → %s   [%s]"
                    % (str(o.get("site", "?")).upper(),
                       o.get("telescope", "?"),
                       short(o.get("start")), short(o.get("end")),
                       o.get("state", "?")))
            if ncanceled:
                say("       (and %d cancelled — routine rescheduling, "
                    "not a failure)" % ncanceled)
            if not obs:
                if ncanceled:
                    say("     no active observation at the moment.")
                else:
                    say("     the scheduler has never created a block for "
                        "this request.")
    say("-" * 72)


def main():
    token = clean_token(os.environ.get("LCO_TOKEN", ""))
    if token and token_problem(token):
        print("The token in the LCO_TOKEN environment variable is not valid: "
              + token_problem(token))
        token = ""
    if not token:
        token = ask_token()
    proposal = ask_proposal()
    print()

    report_allocation(token, proposal)
    report_groups(token, proposal)

    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    path = os.path.join(here, REPORT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_buf.getvalue())
    print("\nThe report was also saved to %s" % path)
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
