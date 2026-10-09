#!/usr/bin/env python3
"""grazr's `claude` shim, first on the PATH of every Herdr pane.

`grazr.py pins-install` copies this to <state>/bin/claude, and pins.py,
stores.py and atomic.py to <state>/lib, so it runs without the plugin's own
directory, which moves with every update. A shell block puts <state>/bin first
on PATH only where HERDR_PANE_ID is set.

In a pane pinned to an account it starts the real Claude on that account's
token. Anywhere else, or on any trouble at all, it starts the real Claude
unchanged: the shim must never be the reason Claude does not start.
"""

import getpass
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.dirname(HERE)


def real_claude(path):
    """The first `claude` on PATH that is not this shim."""
    for directory in path.split(os.pathsep):
        if not directory or os.path.abspath(directory) == HERE:
            continue
        candidate = os.path.join(directory, "claude")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def pinned_env(environ, state=STATE, read_token=None, clock=time.time):
    """The environment to start Claude with, and a warning or None. Only a pane
    pinned to an account with a current token changes anything."""
    pane = environ.get("HERDR_PANE_ID")
    if not pane:
        return dict(environ), None
    sys.path.insert(0, os.path.join(state, "lib"))
    import pins

    # A token already in the environment was put there on purpose: by you, or
    # by the pinned Claude this one runs under.
    if environ.get(pins.ENV_TOKEN):
        return dict(environ), None
    account = (pins.load(state).get(pane) or {}).get("account")
    if not account:
        pins.started(state, pane, None)
        return dict(environ), None
    expires = (pins.tokens(state).get(account) or {}).get("expires_at")
    token = None
    if not (isinstance(expires, (int, float)) and expires <= clock()):
        token = (read_token or _read_token)(state, account)
    if not token:
        pins.started(state, pane, None)
        return dict(environ), (
            "grazr: this pane is pinned to an account with no current token, "
            "so Claude runs on the shared account. Set the token up again in the Accounts window"
        )
    pins.started(state, pane, account)
    return dict(environ, **{pins.ENV_TOKEN: token, pins.ENV_PIN: account}), None


def _read_token(state, account):
    import stores

    if stores.IS_MAC:
        store = stores.KeychainStore(None, getpass.getuser())
    else:
        store = stores.FileStore(None, os.path.join(state, "credentials"))
    token = store.read_token(account)
    return token.strip() if token else None


def main(argv):
    real = real_claude(os.environ.get("PATH", ""))
    if real is None:
        print("grazr: no claude on PATH besides this shim", file=sys.stderr)
        return 127
    try:
        environ, warning = pinned_env(os.environ)
    except Exception as error:  # noqa: BLE001 -- Claude starts whatever happened here
        environ, warning = dict(os.environ), "grazr: the pin could not be read (%s)" % type(error).__name__
    if warning:
        print(warning, file=sys.stderr)
    os.execve(real, [real] + argv[1:], environ)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
