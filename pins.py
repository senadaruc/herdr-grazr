"""Agents pinned to an account.

A pin binds one Herdr pane to one enrolled account. Claude in that pane runs on
the account's own long-lived token (`claude setup-token`) instead of the shared
login grazr rotates, so the rotation never moves it. The pin is keyed by the
pane id, which Herdr keeps across a server restart; the `claude` shim on the
pane's PATH looks it up every time Claude starts there, a resume included.

pins.json holds no secret: pane ids, account ids and times. The tokens live in
the credential store, next to the parked logins.
"""

import contextlib
import fcntl
import json
import os
from datetime import timedelta

import atomic

PINS = "pins.json"          # {pane id: {"account": id or null, "running": id or null, "at": unix}}
PINS_LOCK = "pins.lock"
TOKENS = "tokens.json"      # {account id: {"set_at": unix, "expires_at": unix}}

# What the shim hands the Claude it starts, and so every hook and status line
# that Claude runs: the account this pane is pinned to.
ENV_PIN = "GRAZR_PIN"
ENV_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"

# `claude setup-token` makes a token that lasts a year.
TOKEN_LIFETIME = timedelta(days=365)

EXCLUDE = "exclude"
KEEP = "keep"
ROTATION_MODES = (EXCLUDE, KEEP)

# How the `$grazr` tag marks a pinned pane, so a reader can tell it from the
# shared rotation's account.
PINNED_SUFFIX = " · pinned"


@contextlib.contextmanager
def _locked(state_dir):
    with open(os.path.join(state_dir, PINS_LOCK), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def load(state_dir):
    """Every pin, as {pane id: {"account", "running", "at"}}. A missing or
    unreadable file is no pins: a pin never stops Claude from starting."""
    try:
        with open(os.path.join(state_dir, PINS)) as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {
        pane: entry for pane, entry in stored.items()
        if isinstance(pane, str) and isinstance(entry, dict)
    }


def _save(state_dir, pins):
    atomic.write(os.path.join(state_dir, PINS), json.dumps(pins, sort_keys=True))


def pin(state_dir, pane_id, account_id, now):
    """Pin `pane_id` to `account_id`. What Claude there runs on now is left
    alone: the pin applies when Claude next starts in the pane."""
    with _locked(state_dir):
        pins = load(state_dir)
        entry = dict(pins.get(pane_id) or {})
        entry.update(account=account_id, at=now.timestamp())
        entry.setdefault("running", None)
        pins[pane_id] = entry
        _save(state_dir, pins)


def unpin(state_dir, pane_id, now):
    """Drop the pin. A Claude already running pinned keeps its token until it
    exits, so the entry stays, emptied, until the shim next starts Claude there
    or the pane goes away. Returns whether the pane was pinned."""
    with _locked(state_dir):
        pins = load(state_dir)
        entry = pins.get(pane_id)
        if not entry or not entry.get("account"):
            return False
        if entry.get("running"):
            pins[pane_id] = dict(entry, account=None, at=now.timestamp())
        else:
            del pins[pane_id]
        _save(state_dir, pins)
        return True


def started(state_dir, pane_id, account_id):
    """The shim started Claude in `pane_id` on `account_id`, or on the shared
    login when None. An unpinned pane's leftover entry goes."""
    with _locked(state_dir):
        pins = load(state_dir)
        entry = pins.get(pane_id)
        if entry is None and account_id is None:
            return
        if entry is not None and not entry.get("account") and account_id is None:
            del pins[pane_id]
        else:
            pins[pane_id] = dict(entry or {}, running=account_id)
        _save(state_dir, pins)


def prune(state_dir, live_panes):
    """Forget pins on panes that no longer exist. `live_panes` None means Herdr
    could not say, and then nothing is forgotten. Returns the panes dropped."""
    if live_panes is None:
        return []
    live = set(live_panes)
    with _locked(state_dir):
        pins = load(state_dir)
        gone = [pane for pane in pins if pane not in live]
        if gone:
            _save(state_dir, {pane: entry for pane, entry in pins.items() if pane in live})
        return gone


def held(state_dir):
    """Accounts a pane is pinned to or still runs pinned on."""
    accounts = set()
    for entry in load(state_dir).values():
        for key in ("account", "running"):
            if isinstance(entry.get(key), str) and entry[key]:
                accounts.add(entry[key])
    return accounts


def running_on(state_dir):
    """{pane id: account id} for the panes Claude runs pinned in right now."""
    return {
        pane: entry["running"] for pane, entry in load(state_dir).items()
        if isinstance(entry.get("running"), str) and entry["running"]
    }


def label(name):
    return name + PINNED_SUFFIX


# Token bookkeeping. The token itself is in the credential store; this keeps
# only when it was set and when it lapses, which the Accounts window shows.

def tokens(state_dir):
    try:
        with open(os.path.join(state_dir, TOKENS)) as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return {}
    return stored if isinstance(stored, dict) else {}


def record_token(state_dir, account_id, now):
    with _locked(state_dir):
        stored = tokens(state_dir)
        stored[account_id] = {
            "set_at": now.timestamp(),
            "expires_at": (now + TOKEN_LIFETIME).timestamp(),
        }
        atomic.write(os.path.join(state_dir, TOKENS), json.dumps(stored, sort_keys=True))


def forget_token(state_dir, account_id):
    with _locked(state_dir):
        stored = tokens(state_dir)
        if stored.pop(account_id, None) is not None:
            atomic.write(os.path.join(state_dir, TOKENS), json.dumps(stored, sort_keys=True))


def token_expired(state_dir, account_id, now):
    expires = (tokens(state_dir).get(account_id) or {}).get("expires_at")
    return isinstance(expires, (int, float)) and expires <= now.timestamp()
