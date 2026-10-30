import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import time
import urllib.error
import urllib.request
from collections import namedtuple
from datetime import datetime, timezone

import accounts
import atomic
import core

# proper-lockfile options Claude uses for .oauth_refresh.lock in the same binary.
OAUTH_LOCK_STALE_MS = 60_000

# Claude's saveConfigWithLock builds this path at runtime, so the name is not
# greppable. Its stale age is unpublished, so this matches the other lock.
CONFIG_LOCK_STALE_MS = 60_000

# `mcpOAuth` holds a login per MCP server, minted against that server and not
# against a Claude account, so it survives a swap. An allowlist, not "everything
# except the login": a key Claude adds later must not start crossing accounts.
CARRIED_CREDENTIAL_KEYS = ("mcpOAuth",)

# `primaryApiKey` is a raw key Claude reads back as an auth source, so a
# leftover spends the outgoing account's key on the incoming one. The caches
# describe its plan and model access. Claude re-fetches each on demand, so
# dropping them costs a request. `cachedUsageUtilization` is left in place:
# read_limits already discards it when it names another account.
ACCOUNT_SCOPED_CONFIG_KEYS = (
    "primaryApiKey",
    "customApiKeyResponses",
    "overageCreditGrantCache",
    "passesEligibilityCache",
    "passesLastSeenRemaining",
    "cachedExtraUsageDisabledReason",
    "orgModelDefaultCache",
    "modelAccessCache",
    "additionalModelCostsCache",
    "additionalModelOptionsCache",
)


@contextlib.contextmanager
def _proper_lock(path, stale_ms, busy_message):
    """Mirrors proper-lockfile, so grazr and Claude exclude each other: acquire
    by mkdir, treat a holder past its stale age as abandoned, and on release
    leave behind any lock whose mtime changed, since that one is Claude's.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        try:
            age_ms = (time.time() - os.path.getmtime(path)) * 1000
        except FileNotFoundError:
            # Released between the two calls. There is no holder to respect,
            # so fall through to the retake below rather than raising.
            age_ms = stale_ms
        if age_ms < stale_ms:
            raise RuntimeError(busy_message)
        # Another process may be breaking the same lock. The loser must back
        # off rather than assume it holds it too.
        try:
            os.rmdir(path)
        except FileNotFoundError:
            pass
        try:
            os.mkdir(path)
        except FileExistsError:
            raise RuntimeError(busy_message)
    held = [os.path.getmtime(path)]

    def renew():
        """Claude refreshes its own lock every 5s. Slow work must do the same,
        or the lock looks abandoned and Claude takes it."""
        os.utime(path, None)
        held[0] = os.path.getmtime(path)

    try:
        yield renew
    finally:
        held_since = held[0]
        try:
            if os.path.getmtime(path) == held_since:
                os.rmdir(path)
        except OSError:
            pass


def _oauth_refresh_lock(config_dir):
    """Held across a swap so a token refresh cannot rewrite the live item."""
    return _proper_lock(
        os.path.join(config_dir, ".oauth_refresh.lock"),
        OAUTH_LOCK_STALE_MS,
        "Claude is refreshing its token, so grazr is not swapping now",
    )


Paths = namedtuple("Paths", "config_path config_dir accounts_dir")


STATUSLINE_WINDOWS = (("five_hour", "session", "session"), ("seven_day", "weekly_all", "weekly"))


def statusline_limits(payload):
    """The limits Claude hands its status-line command after every message, for
    the token the session is using. None when the payload has none."""
    try:
        windows = json.loads(payload)["rate_limits"]
        return [
            core.Limit(
                kind=kind,
                scope=None,
                group=group,
                remaining=max(0, 100 - windows[window]["used_percentage"]),
                resets_at=_from_unix(windows[window].get("resets_at")),
            )
            for window, kind, group in STATUSLINE_WINDOWS
            if window in windows
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


# Per-model weekly limits, for MODEL_LIMITS=1. The status line carries only the
# five-hour and all-models weekly windows, so a per-model cap runs out with no
# reading at all. The usage endpoint behind `/usage` does report it, as a
# `weekly_scoped` entry naming the model. It is asked for the live login only,
# and its token is never refreshed here: an expired one means no reading this
# time, and Claude refreshes it on its next request.
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
USAGE_TIMEOUT_SECONDS = 10


def access_token(blob):
    """The live login's access token, or None when the blob is not a claude.ai
    login (an API key, a test double, nothing at all)."""
    try:
        token = json.loads(blob)["claudeAiOauth"]["accessToken"]
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    return token if isinstance(token, str) and token else None


def scoped_limits(reply):
    """Per-model limits from a usage reply, as Limits whose scope is the model's
    display name. Unscoped entries are left to the status line, which is fresher.
    None when the reply has no readable `limits`."""
    try:
        entries = reply["limits"]
        if not isinstance(entries, list):
            return None
    except (TypeError, KeyError):
        return None
    limits = []
    for entry in entries:
        try:
            model = ((entry.get("scope") or {}).get("model") or {}).get("display_name")
            percent = entry["percent"]
        except (AttributeError, KeyError, TypeError):
            continue
        if not isinstance(model, str) or not model:
            continue
        if not isinstance(percent, (int, float)) or isinstance(percent, bool):
            continue
        try:
            resets_at = accounts.parse_time(entry.get("resets_at"))
        except (TypeError, ValueError, AttributeError):
            resets_at = None
        limits.append(core.Limit(
            kind=entry.get("kind") or "weekly_scoped",
            scope=model,
            group=entry.get("group") or "weekly",
            remaining=max(0, 100 - percent),
            resets_at=resets_at,
        ))
    return limits


USAGE_USER_AGENT = "grazr (+https://github.com/wazum/herdr-grazr)"


def fetch_scoped_limits(blob, opener=urllib.request.urlopen):
    """Ask the usage endpoint for the login in `blob`. None on any failure: no
    token, an expired one, the network, an unexpected shape."""
    reply = _fetch_usage(blob, opener)
    return None if reply is None else scoped_limits(reply)


def fetch_usage_limits(blob, opener=urllib.request.urlopen):
    """Every window the usage endpoint reports for the login in `blob`, the
    five-hour and all-models ones included: a parked account has no status
    line to give them. None on any failure, like fetch_scoped_limits."""
    reply = _fetch_usage(blob, opener)
    return None if reply is None else usage_limits(reply)


PROFILE_URL = "https://api.anthropic.com/api/oauth/profile"


def token_account(token, opener=urllib.request.urlopen):
    """{"uuid", "email"} of the account a pin token belongs to, or None when
    Claude will not say: a setup-token may lack the profile scope."""
    request = urllib.request.Request(PROFILE_URL, headers={
        "Authorization": "Bearer " + token,
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": USAGE_USER_AGENT,
    })
    try:
        with opener(request, timeout=USAGE_TIMEOUT_SECONDS) as response:
            account = json.loads(response.read().decode("utf-8"))["account"]
        return {"uuid": account["uuid"], "email": account.get("email") or account.get("email_address")}
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError, KeyError, TypeError):
        return None


def _fetch_usage(blob, opener):
    token = access_token(blob)
    if token is None:
        return None
    request = urllib.request.Request(USAGE_URL, headers={
        "Authorization": "Bearer " + token,
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": USAGE_USER_AGENT,
    })
    try:
        with opener(request, timeout=USAGE_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError):
        return None


def usage_limits(reply):
    """Every window in a usage reply: per-model ones as scoped_limits reads
    them, the rest with no scope. None when the reply has no readable `limits`."""
    scoped = scoped_limits(reply)
    if scoped is None:
        return None
    unscoped = []
    for entry in reply["limits"]:
        try:
            if entry.get("scope"):
                continue
            kind, group, percent = entry["kind"], entry["group"], entry["percent"]
        except (AttributeError, KeyError, TypeError):
            continue
        if not isinstance(kind, str) or not isinstance(group, str):
            continue
        if not isinstance(percent, (int, float)) or isinstance(percent, bool):
            continue
        try:
            resets_at = accounts.parse_time(entry.get("resets_at"))
        except (TypeError, ValueError, AttributeError):
            resets_at = None
        unscoped.append(core.Limit(
            kind=kind, scope=None, group=group, remaining=max(0, 100 - percent), resets_at=resets_at,
        ))
    return unscoped + scoped


# PARKED_POLL_MINUTES: a parked account's own token, refreshed the way Claude
# Code refreshes it (its 2.1.x client, endpoint and request body), so grazr can
# read that account's usage while nothing runs on it. Only parked logins: the
# live one belongs to Claude, which refreshes it itself.
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_TIMEOUT_SECONDS = 30
# Refreshed this long before it lapses, so the usage request never races it.
TOKEN_EXPIRY_MARGIN_MS = 5 * 60 * 1000


def token_expired(blob, now_ms):
    """Whether the login's access token has lapsed, or is about to. A blob with
    no expiry on record reads as current: the request will say otherwise."""
    try:
        expires_at = json.loads(blob)["claudeAiOauth"].get("expiresAt")
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    expires_at = _epoch_milliseconds(expires_at)
    return expires_at is not None and expires_at - TOKEN_EXPIRY_MARGIN_MS <= now_ms


def refreshed_login(blob, opener=urllib.request.urlopen, now_ms=None):
    """`blob` with a new token pair, or None when there is no refresh token or
    the server refuses. The refresh token rotates: once this returns, the old
    one is spent, so the caller must store the result before anything else
    reads the login."""
    try:
        login = json.loads(blob)
        oauth = dict(login["claudeAiOauth"])
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    refresh_token = oauth.get("refreshToken")
    if not isinstance(refresh_token, str) or not refresh_token:
        return None
    body = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID}
    scopes = oauth.get("scopes")
    if isinstance(scopes, list) and scopes and all(isinstance(scope, str) for scope in scopes):
        body["scope"] = " ".join(scopes)
    request = urllib.request.Request(
        TOKEN_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USAGE_USER_AGENT},
        method="POST",
    )
    try:
        with opener(request, timeout=TOKEN_TIMEOUT_SECONDS) as response:
            reply = json.loads(response.read().decode("utf-8"))
        access = reply["access_token"]
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError, KeyError, TypeError):
        return None
    if not isinstance(access, str) or not access:
        return None
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    oauth["accessToken"] = access
    if isinstance(reply.get("refresh_token"), str) and reply["refresh_token"]:
        oauth["refreshToken"] = reply["refresh_token"]
    expires_in = _epoch_milliseconds(reply.get("expires_in"))
    if expires_in is not None:
        oauth["expiresAt"] = int(now_ms + expires_in * 1000)
    refresh_expires_in = _epoch_milliseconds(reply.get("refresh_token_expires_in"))
    if refresh_expires_in is not None:
        oauth["refreshTokenExpiresAt"] = int(now_ms + refresh_expires_in * 1000)
    login["claudeAiOauth"] = oauth
    return json.dumps(login)


def _from_unix(seconds):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return None
    return datetime.fromtimestamp(seconds, timezone.utc)


def active_account(paths):
    try:
        with open(paths.config_path) as handle:
            return (json.load(handle).get("oauthAccount") or {}).get("accountUuid") or None
    except (OSError, ValueError, AttributeError):
        return None


def enrol(paths, store, name, source_config_dir=None, expect_email=None):
    """Copy an existing login into grazr's keeping. Authentication is always
    Claude's own `claude auth login`; grazr never sees a password. `source_config_dir`
    is an isolated CLAUDE_CONFIG_DIR, so enrolling a second account never
    disturbs the live one."""
    config_path = (
        os.path.join(source_config_dir, ".claude.json")
        if source_config_dir
        else paths.config_path
    )
    blob = (
        store.read_isolated(source_config_dir) if source_config_dir else store.read_live()
    )
    if blob is None:
        raise RuntimeError("No login found. Run `claude auth login` first")

    with open(config_path) as handle:
        identity = (json.load(handle).get("oauthAccount") or {})
    return enrol_identity(paths, store, name, identity, blob, expect_email)


def enrol_identity(paths, store, name, identity, blob, expect_email=None):
    """Park a credential blob and record its account, given the identity it
    belongs to. Shared by a login read from a config dir and one imported as
    JSON, so both leave identical records. Nothing is written when it refuses.
    `expect_email` is compared case-insensitively: a login under another
    address is refused rather than enrolled under the wrong account."""
    account_id = accounts.account_id(identity.get("accountUuid"))
    # A non-string e-mail or organisation is stored as None rather than as whatever was sent.
    identity = dict(identity)
    for key in ("emailAddress", "organizationName"):
        if not isinstance(identity.get(key), str):
            identity[key] = None

    if expect_email is not None:
        signed_in = str(identity.get("emailAddress") or "")
        if signed_in.lower() != expect_email.lower():
            raise RuntimeError(
                "signed in as %s, expected %s"
                % (signed_in or "an account with no e-mail", expect_email)
            )

    # Accounts are looked up by name, so a duplicate hides one of them for good.
    for existing in accounts.load(paths, []):
        if existing.name == name and existing.id != account_id:
            raise RuntimeError("The name %r is already used by another account" % name)

    store.write_parked(account_id, _parkable(blob))
    accounts.write(
        paths,
        account_id,
        {
            "name": name,
            "email": identity.get("emailAddress"),
            "organization": identity.get("organizationName"),
            "oauthAccount": identity,
            "snapshot": None,
        },
    )
    return account_id


def export_isolated_login(store, config_dir):
    """The credential blob and identity of an isolated login, for another
    engine to import. The blob is whatever the store returns and is only ever
    handed back to the caller; nothing here prints or logs it."""
    blob = store.read_isolated(config_dir)
    try:
        with open(os.path.join(config_dir, ".claude.json")) as handle:
            identity = (json.load(handle).get("oauthAccount") or {})
    except (OSError, ValueError, AttributeError):
        identity = {}
    if not isinstance(identity, dict):
        identity = {}
    if blob is None or not identity.get("accountUuid"):
        raise RuntimeError("No login found in %s. Run `claude auth login` there first" % config_dir)
    return {"identity": identity, "blob": blob}


def discard_isolated_login(store, config_dir):
    """Remove the throwaway login enrolment used, stored credential and all.
    The copy grazr keeps is the parked one; this one is an untracked duplicate.

    Returns whether the credential is gone. Never raises: it is called from a
    finally, and masking the failure it is cleaning up after would hide the
    real problem.
    """
    removed = store.discard_isolated(config_dir)
    shutil.rmtree(config_dir, ignore_errors=True)
    return removed


def rotate(paths, store, active_id, next_id, snapshot):
    """Park the live credential under the account leaving, install the one
    arriving, and move the identity with it. Every refusal happens before the
    first write, and a failure part-way is resumed by the next attempt rather
    than parked over."""
    override = settings_auth_override(paths.config_dir)
    if override:
        raise RuntimeError(
            "%s in settings.json puts Claude on API-key auth, so swapping the "
            "saved claude.ai login would change nothing. Unset it to let grazr "
            "rotate" % override
        )

    arriving = store.read_parked(next_id)
    if arriving is None:
        raise RuntimeError("No parked credential for %s. Enrol it again" % next_id)
    identity = accounts.read(paths, next_id)["oauthAccount"]

    # Both locks before the first write. The identity write comes last, but its
    # lock can be busy too, and refusing after the credential moved is the whole
    # thing this order prevents.
    with _config_lock(paths.config_path) as renew_config, _oauth_refresh_lock(
        paths.config_dir
    ) as renew_oauth:

        def renew():
            renew_config()
            renew_oauth()

        leaving = store.read_live()
        if leaving is None:
            raise RuntimeError("There is no live credential to park, so grazr is not swapping")
        renew()
        pending = _pending_swap(paths, store, active_id, leaving)
        if pending is None:
            parking = _parkable(leaving)
            # The marker goes first, so a retry can tell how far this got.
            _write_swap_marker(paths, active_id, next_id, parking)
            store.write_parked(active_id, parking)
            renew()
            try:
                store.write_live(_carry_shared_keys(arriving, leaving))
            except ValueError:
                # The store refuses a blob past its line length, and the carry
                # is what pushes it over. Losing the MCP logins beats leaving
                # the pane on a spent account.
                store.write_live(arriving)
            renew()
        elif pending["next"] != next_id:
            _merge_oauth_account(
                paths.config_path, accounts.read(paths, pending["next"])["oauthAccount"]
            )
            _clear_swap_marker(paths)
            raise RuntimeError(
                "Finished a swap to %s that had been interrupted. Try again" % pending["next"]
            )
        _merge_oauth_account(paths.config_path, identity)
        _clear_swap_marker(paths)

    accounts.record_snapshot(paths, active_id, snapshot)
    return _arrival_note(arriving)


def _arrival_note(blob):
    """What to say about the credential a swap just installed, or None.

    Claude refreshes an expired token on its next request, so a lapse is not a
    refusal. A refresh that fails signs the account out, and then these two
    times are the only record of what grazr handed over.
    """
    try:
        oauth = json.loads(blob)["claudeAiOauth"]
        expires_at = _epoch_milliseconds(oauth.get("expiresAt"))
        refresh_expires_at = _epoch_milliseconds(oauth.get("refreshTokenExpiresAt"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    notes = []
    if expires_at is not None and expires_at <= time.time() * 1000:
        notes.append("Its token had expired while parked, so Claude has to refresh it")
    if refresh_expires_at is not None:
        notes.append(
            "The refresh behind it runs to %s"
            % datetime.fromtimestamp(refresh_expires_at / 1000, timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
    return ". " + ". ".join(notes) if notes else None


def _epoch_milliseconds(value):
    """Claude writes its token times as epoch milliseconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return value


def settings_auth_override(config_dir):
    """The setting in settings.json that puts Claude on API-key auth, or None.

    Claude reaches every one of these before the saved claude.ai login, so a
    swap underneath one changes nothing while the log says it worked. Its own
    words in the 2.1.261 binary: "ANTHROPIC_API_KEY is set, so this session is
    using API-key auth", and the same for the other two. `env` there is
    "Environment variables to set for Claude Code sessions", so a key parked in
    that block counts as set.
    """
    try:
        with open(os.path.join(config_dir, "settings.json")) as handle:
            settings = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(settings, dict):
        return None
    if str(settings.get("apiKeyHelper") or "").strip():
        return "apiKeyHelper"
    env = settings.get("env")
    if isinstance(env, dict):
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if str(env.get(key) or "").strip():
                return "env.%s" % key
    return None


# Herdr's and Orca's grazr can share one settings.json. The state dir a shim
# carries is what says which of them installed it.
STATE_DIR_VARS = ("GRAZR_STATE_DIR", "HERDR_PLUGIN_STATE_DIR")
SHIM_VARS = STATE_DIR_VARS + ("GRAZR_CONFIG_DIR", "HERDR_PLUGIN_CONFIG_DIR", "GRAZR_NOTIFIER")

# The record grazr.py keeps in each state dir, of the bar it displaced.
PREVIOUS_STATUSLINE = "statusline.previous.json"


def install_statusline(config_dir, record_path, command, force=False):
    """Put grazr's shim in as Claude's status line and keep what was there.

    The record keeps the exact shim too, so a plugin checked out under a new
    path re-installs instead of passing as connected, and a status line the
    user replaced by hand is recognised as theirs. Another grazr's shim is
    refused unless force, which takes it over and keeps the bar that grazr
    had displaced.
    """
    settings = _read_settings(config_dir)
    current = settings.get("statusLine")
    if isinstance(current, dict) and current.get("command") == command:
        return "Status line already connected"
    recorded = _read_record(record_path)
    if _is_ours(current, command):
        # Ours from an older checkout path, or ours with the record lost. Only
        # the record knows what it displaced, so without one there is nothing
        # to restore.
        previous = recorded["previous"] if recorded else None
    elif is_shim(current):
        other = _state_label(carried_state_dir(current))
        if not force:
            raise RuntimeError("Claude's status line already belongs to another grazr (state in %s); disconnect it there, or run install --force to take it over" % other)
        # Its shim is never chained as previous: _previous_bar refuses to run a
        # bar that runs grazr. The bar it kept is what comes back instead.
        previous = _foreign_previous(carried_state_dir(current))
    elif runs_grazr(current):
        # The user's own bar calls grazr for its segment. Wrapping it would
        # make grazr run the bar, which runs grazr, on every message.
        return "Status line already runs grazr in your own bar, left alone"
    else:
        previous = current
    atomic.write(record_path, json.dumps({"previous": previous, "shim": command}))
    settings["statusLine"] = {
        "type": "command",
        "command": command,
        "refreshInterval": (previous or {}).get("refreshInterval", 60),
    }
    _write_settings(config_dir, settings)
    return "Status line connected"


def uninstall_statusline(config_dir, record_path):
    recorded = _read_record(record_path)
    if recorded is None:
        return "Status line was not connected"
    settings = _read_settings(config_dir)
    if not _is_ours(settings.get("statusLine"), recorded["shim"]):
        return "Status line was not grazr's, left alone"
    if recorded["previous"]:
        settings["statusLine"] = recorded["previous"]
    else:
        del settings["statusLine"]
    _write_settings(config_dir, settings)
    os.remove(record_path)
    return "Status line disconnected"


def statusline_installed(config_dir, record_path, state_dir=None):
    """grazr's shim is in, or the user's own bar runs grazr: either way
    every message reaches it. Another grazr's shim, or a bar that runs only
    another grazr, does not count.

    With no record, a bar counts only when it runs state_dir, the one grazr
    is installed with, since nothing else says whose bar it is. Without
    state_dir a bar counts as it always has."""
    try:
        current = _read_settings(config_dir).get("statusLine")
    except RuntimeError:
        return False
    recorded = _read_record(record_path)
    if is_shim(current):
        return recorded is not None and same_owner(current, recorded["shim"])
    if runs_grazr(current):
        if recorded is None:
            return state_dir is None or _normal_dir(state_dir) in _named_state_dirs(current)
        embedded = _embedded_state_dirs(current)
        return not embedded or _normal_dir(carried_state_dir(recorded["shim"])) in _named_state_dirs(current)
    return False


def runs_grazr(entry):
    """A status line, or its command, that invokes grazr's own status line
    somewhere in it, whether it is the shim or a bar of the user's that
    calls grazr for one segment."""
    command = entry.get("command", "") if isinstance(entry, dict) else (entry or "")
    return "grazr.py" in command and "statusline" in command


def shim_parts(command):
    """(entry, state_dir) when command is grazr's own shim, else None. A shim is
    NAME=value pairs from SHIM_VARS, python, a grazr.py path, then statusline or
    failure, and nothing more. state_dir is "" when the shim carries none."""
    if isinstance(command, dict):
        command = command.get("command")
    if not isinstance(command, str):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    state_dir = ""
    while words and "=" in words[0] and words[0].split("=", 1)[0] in SHIM_VARS:
        name, value = words.pop(0).split("=", 1)
        if name in STATE_DIR_VARS:
            state_dir = value
    if len(words) != 3 or not words[0].endswith(("python3", "python")):
        return None
    if not words[1].endswith("grazr.py") or words[2] not in ("statusline", "failure"):
        return None
    return words[2], state_dir


def is_shim(command):
    return shim_parts(command) is not None


def carried_state_dir(command):
    """The state dir a shim carries, "" for a shim that carries none, and None
    when command is not a shim at all."""
    parts = shim_parts(command)
    return None if parts is None else parts[1]


def same_owner(a, b):
    """Both are grazr shims, and they carry the same state dir once spelled
    the same way, so "/s/", "$HOME/s" and "~/s" are one owner as "/s" is."""
    if not (is_shim(a) and is_shim(b)):
        return False
    return _normal_dir(carried_state_dir(a)) == _normal_dir(carried_state_dir(b))


def _normal_dir(state_dir):
    """One spelling of a state dir for comparisons: $VAR and ~ expanded,
    trailing slashes and ./ dropped. The empty string is the default ~/.grazr,
    which is where a shim that carries no state dir runs."""
    return os.path.normpath(os.path.expanduser(os.path.expandvars(state_dir or "~/.grazr")))


def _named_state_dirs(entry):
    """Every state dir a bar names, spelled by _normal_dir, with the default
    when it names none."""
    return [_normal_dir(value) for value in _embedded_state_dirs(entry)] or [_normal_dir("")]


def _is_ours(entry, shim):
    return same_owner(entry, shim)


def _current_statusline(config_dir):
    """Claude's status line, or None when settings.json cannot be read. Reads
    that fail are reported where they are touched, not here."""
    try:
        return _read_settings(config_dir).get("statusLine")
    except RuntimeError:
        return None


def foreign_statusline(config_dir, shim):
    """The state dir of another grazr that holds Claude's status line, or None.
    That is another grazr's shim, or a bar of the user's that calls one. A shim
    that carries no state dir, or a bar that names none, is reported as ~/.grazr."""
    current = _current_statusline(config_dir)
    if is_shim(current):
        return None if same_owner(current, shim) else _state_label(carried_state_dir(current))
    return _foreign_in_bar(current, shim)


def foreign_bar(config_dir, shim):
    """The state dir of another grazr that the user's own status line calls, or
    None. install refuses that whatever force says: taking the hook while the
    bar stays would leave two grazrs answering the same messages."""
    return _foreign_in_bar(_current_statusline(config_dir), shim)


def _foreign_in_bar(entry, shim):
    """The first state dir a user's bar names that is not shim's, or None when
    the bar is not one that runs grazr, is a shim, or names only shim's dir."""
    if is_shim(entry) or not runs_grazr(entry):
        return None
    ours = _normal_dir(carried_state_dir(shim))
    for state_dir in _embedded_state_dirs(entry) or [""]:
        if _normal_dir(state_dir) != ours:
            return _state_label(state_dir)
    return None


def _state_label(state_dir):
    return state_dir or "~/.grazr"


def _foreign_previous(state_dir):
    """The bar another grazr kept in its own record, or None when that cannot
    be read or would run grazr. The empty state dir means ~/.grazr."""
    path = os.path.join(state_dir or os.path.expanduser("~/.grazr"), PREVIOUS_STATUSLINE)
    try:
        with open(path) as handle:
            previous = json.load(handle).get("previous")
    except (OSError, ValueError, AttributeError):
        return None
    return previous if isinstance(previous, dict) and not runs_grazr(previous) else None


def _embedded_state_dirs(entry):
    """The state dirs a bar names in its own command. A bar of the user's that
    calls grazr for one segment is told apart by whose state dir it names."""
    command = entry.get("command", "") if isinstance(entry, dict) else (entry or "")
    pattern = r"(?:%s)=(\"[^\"]*\"|'[^']*'|\S+)" % "|".join(STATE_DIR_VARS)
    return [_unquote(value) for value in re.findall(pattern, command)]


def _unquote(value):
    try:
        words = shlex.split(value)
    except ValueError:
        return value
    return words[0] if words else ""


# The StopFailure errors that say the account itself is the problem. Claude
# matches the hook's `matcher` against the `error` field as a regex.
FAILURE_HOOK_EVENT = "StopFailure"
FAILURE_HOOK_MATCHER = (
    "oauth_org_not_allowed|billing_error|account_on_hold|authentication_failed|rate_limit"
)


def install_failure_hook(config_dir, command, force=False):
    """Add grazr's StopFailure hook, next to whatever hooks are there. A grazr
    hook from an older checkout is replaced rather than doubled. Another
    grazr's group is refused unless force, which takes it over. Hooks that are
    not grazr's are always kept."""
    settings = _read_settings(config_dir)
    hooks = settings.get("hooks")
    if hooks is None:
        hooks = settings["hooks"] = {}
    if not isinstance(hooks, dict):
        raise RuntimeError("settings.json has a \"hooks\" that is not an object, not touching it")
    groups = hooks.get(FAILURE_HOOK_EVENT) or []
    if not isinstance(groups, list):
        raise RuntimeError("settings.json has a %s that is not a list, not touching it" % FAILURE_HOOK_EVENT)
    wanted = {"matcher": FAILURE_HOOK_MATCHER, "hooks": [{"type": "command", "command": command}]}
    if wanted in groups:
        return "Failure hook already connected"
    foreign = [group for group in groups if _is_foreign_failure_group(group, command)]
    if foreign and not force:
        raise RuntimeError("Claude's StopFailure hook already belongs to another grazr (state in %s); disconnect it there, or run install --force to take it over" % _group_state(foreign[0]))
    hooks[FAILURE_HOOK_EVENT] = [group for group in groups if _grazr_failure_state(group) is None] + [wanted]
    _write_settings(config_dir, settings)
    return "Failure hook connected"


def uninstall_failure_hook(config_dir, command):
    settings = _read_settings(config_dir)
    hooks = settings.get("hooks")
    groups = hooks.get(FAILURE_HOOK_EVENT) if isinstance(hooks, dict) else None
    if not isinstance(groups, list) or not any(_is_our_failure_group(group, command) for group in groups):
        return "Failure hook was not connected"
    kept = [group for group in groups if not _is_our_failure_group(group, command)]
    if kept:
        hooks[FAILURE_HOOK_EVENT] = kept
    else:
        del hooks[FAILURE_HOOK_EVENT]
    if not hooks:
        del settings["hooks"]
    _write_settings(config_dir, settings)
    return "Failure hook disconnected"


def failure_hook_installed(config_dir, command):
    """Matched by state dir, not path, so a hook left at a deleted plugin path
    still reads as connected. The status line has always matched the same way."""
    try:
        hooks = _read_settings(config_dir).get("hooks")
    except RuntimeError:
        return False
    groups = hooks.get(FAILURE_HOOK_EVENT) if isinstance(hooks, dict) else None
    return isinstance(groups, list) and any(_is_our_failure_group(group, command) for group in groups)


def foreign_failure_hook(config_dir, failure_command):
    """The state dir of another grazr's StopFailure group, or None."""
    try:
        hooks = _read_settings(config_dir).get("hooks")
    except RuntimeError:
        return None
    groups = hooks.get(FAILURE_HOOK_EVENT) if isinstance(hooks, dict) else None
    for group in groups if isinstance(groups, list) else []:
        if _is_foreign_failure_group(group, failure_command):
            return _group_state(group)
    return None


def foreign_owner(config_dir, shim, failure_command):
    """The state dir of another grazr holding either slot, or None."""
    return foreign_statusline(config_dir, shim) or foreign_failure_hook(config_dir, failure_command)


def _failure_group_command(group):
    """The command of a group grazr wrote, which holds only its own failure
    shim, whatever else the hook says. None for any other group."""
    if not isinstance(group, dict) or not isinstance(group.get("hooks"), list) or len(group["hooks"]) != 1:
        return None
    hook = group["hooks"][0]
    if not isinstance(hook, dict):
        return None
    parts = shim_parts(hook.get("command"))
    return hook["command"] if parts is not None and parts[0] == "failure" else None


def _is_grazr_failure_command(command):
    """A StopFailure command that calls grazr: a shim, or one written by hand."""
    parts = shim_parts(command)
    if parts is not None:
        return parts[0] == "failure"
    return isinstance(command, str) and "grazr.py" in command and command.rstrip().endswith(" failure")


def _grazr_failure_state(group):
    """The state dir of the grazr a StopFailure group belongs to, "" when its
    command carries none, or None when the group is not grazr's. Every hook in
    the group must call grazr. A group that mixes a grazr hook with one that is
    not grazr's is deliberately neither ours nor foreign, so it is left alone.
    """
    hooks = group.get("hooks") if isinstance(group, dict) else None
    if not isinstance(hooks, list) or not hooks:
        return None
    commands = [hook.get("command") if isinstance(hook, dict) else None for hook in hooks]
    if not all(_is_grazr_failure_command(command) for command in commands):
        return None
    state = carried_state_dir(commands[0])
    return "" if state is None else state


def _is_our_failure_group(group, command):
    return same_owner(_failure_group_command(group), command)


def _is_foreign_failure_group(group, command):
    return _grazr_failure_state(group) is not None and not _is_our_failure_group(group, command)


def _group_state(group):
    return _state_label(_grazr_failure_state(group))


def _read_record(record_path):
    try:
        with open(record_path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _read_settings(config_dir):
    try:
        with open(os.path.join(config_dir, "settings.json")) as handle:
            settings = json.load(handle)
    except OSError:
        return {}
    except ValueError:
        raise RuntimeError("%s is not valid JSON, not touching it" % os.path.join(config_dir, "settings.json"))
    return settings if isinstance(settings, dict) else {}


def _write_settings(config_dir, settings):
    os.makedirs(config_dir, exist_ok=True)
    atomic.write(os.path.join(config_dir, "settings.json"), json.dumps(settings, indent=2) + "\n")


SWAP_MARKER = ".swap.json"


def _pending_swap(paths, store, active_id, live):
    """The swap an earlier try left half done, once its credential has moved:
    the retry then only writes the identity, instead of parking the arriving
    credential over the outgoing one. None when there is nothing to resume. A
    live credential that matches neither side is refused, not guessed at."""
    pending = _read_swap_marker(paths)
    if pending is None or pending["active"] != active_id:
        _clear_swap_marker(paths)
        return None
    if _digest(_parkable(live)) == pending["leaving"]:
        _clear_swap_marker(paths)
        return None
    arrived = store.read_parked(pending["next"])
    if arrived is not None and _same_login(live, arrived):
        return pending
    parked = store.read_parked(active_id)
    if parked is None or _digest(parked) != pending["leaving"]:
        # Nothing was parked, so nothing moved. Claude refreshed the outgoing token.
        _clear_swap_marker(paths)
        return None
    raise RuntimeError(
        "A swap from %s to %s was interrupted and the live credential matches "
        "neither account, so grazr leaves it alone. Log in again with `claude auth login`"
        % (active_id, pending["next"])
    )


def _write_swap_marker(paths, active_id, next_id, leaving):
    atomic.write(
        os.path.join(paths.accounts_dir, SWAP_MARKER),
        json.dumps({"active": active_id, "next": next_id, "leaving": _digest(leaving)}),
    )


def _read_swap_marker(paths):
    try:
        with open(os.path.join(paths.accounts_dir, SWAP_MARKER)) as handle:
            marker = json.load(handle)
        return marker if {"active", "next", "leaving"} <= set(marker) else None
    except (OSError, ValueError, TypeError):
        return None


def _clear_swap_marker(paths):
    try:
        os.remove(os.path.join(paths.accounts_dir, SWAP_MARKER))
    except FileNotFoundError:
        pass


def _digest(blob):
    return hashlib.sha256(blob.encode()).hexdigest()


def _same_login(one, other):
    """Same login, whatever MCP logins ride along. A swap carries those onto
    the arriving blob, so the live copy stops matching its parked one."""
    return _without_carried(one) == _without_carried(other)


def _parkable(blob):
    """The credential without the keys a swap carries. A parked copy of them
    is never read, since the carry takes the live ones on the way back in, and
    with enough MCP servers they push the keychain line past its limit. A blob
    with nothing to strip stays byte for byte, like the carry does."""
    stripped = _without_carried(blob)
    if not isinstance(stripped, dict) or stripped == json.loads(blob):
        return blob
    return json.dumps(stripped)


def _without_carried(blob):
    try:
        parsed = json.loads(blob)
    except ValueError:
        return blob
    if not isinstance(parsed, dict):
        return blob
    return {key: value for key, value in parsed.items() if key not in CARRIED_CREDENTIAL_KEYS}


def _carry_shared_keys(arriving, leaving):
    """Move what belongs to no account onto the credential coming in, so a swap
    does not sign the user out of every MCP server.

    Best-effort: a blob that will not parse is installed as it stands. Refusing
    here would strand the pane on an account that has already run out.
    """
    try:
        outgoing, incoming = json.loads(leaving), json.loads(arriving)
    except ValueError:
        return arriving
    if not isinstance(outgoing, dict) or not isinstance(incoming, dict):
        return arriving
    carried = {key: outgoing[key] for key in CARRIED_CREDENTIAL_KEYS if key in outgoing}
    if not carried:
        # Byte for byte, rather than a re-serialisation that only moves whitespace.
        return arriving
    incoming.update(carried)
    return json.dumps(incoming)


def _config_lock(config_path):
    return _proper_lock(
        config_path + ".lock",
        CONFIG_LOCK_STALE_MS,
        "Claude is writing its config, so grazr is not swapping now",
    )


def _merge_oauth_account(config_path, oauth_account):
    """Caller holds the config lock. Claude re-reads and merges under it too,
    so a copy taken before acquiring it could already be out of date."""
    with open(config_path) as handle:
        config = json.load(handle)
    config["oauthAccount"] = oauth_account
    for key in ACCOUNT_SCOPED_CONFIG_KEYS:
        config.pop(key, None)

    atomic.write(config_path, json.dumps(config))


