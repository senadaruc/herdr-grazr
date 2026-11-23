"""The accounts grazr enrolled, as files under its own state directory.

Nothing here is Claude's. An entry holds an identity from a Claude login, but
the file, its name and the parked snapshot are grazr's own.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

import atomic
import core


def parse_time(value):
    """A bare timestamp would raise when compared against an aware `now`, deep
    inside the decision. It lives here rather than in claude.py so this module
    depends on nothing above it."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _canonical_uuid(value):
    """The lowercase canonical form of a uuid, or None for anything else.
    Compared case-insensitively: some providers return uppercase, and it is
    the same account."""
    try:
        canonical = str(uuid.UUID(str(value)))
        if canonical == value.lower():
            return canonical
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def account_id(value):
    """An account id becomes a filename and a keychain service, so it has to be
    a plain uuid, or two of them joined by a dot, and nothing else. The second
    form names a further organization of one login (see new_id)."""
    parts = value.split(".") if isinstance(value, str) else [value]
    canonical = [_canonical_uuid(part) for part in parts]
    if 1 <= len(parts) <= 2 and None not in canonical:
        return ".".join(canonical)
    raise RuntimeError("That login has no usable account identity: %r" % (value,))


def login_of(record):
    """(accountUuid, organizationUuid) of a stored record's `oauthAccount`,
    lowercase, either None when it is missing. A login can belong to several
    organizations, and the pair is what tells their accounts apart."""
    oauth = record.get("oauthAccount") if isinstance(record, dict) else None
    if not isinstance(oauth, dict):
        return None, None
    account, organization = oauth.get("accountUuid"), oauth.get("organizationUuid")
    return (
        account.lower() if isinstance(account, str) and account else None,
        organization.lower() if isinstance(organization, str) and organization else None,
    )


def records(paths):
    """(id, record) for every readable account file, sorted by id."""
    try:
        filenames = sorted(os.listdir(paths.accounts_dir))
    except OSError:
        return []
    found = []
    for filename in filenames:
        if filename.startswith(".") or not filename.endswith(".json"):
            continue
        identifier = filename[: -len(".json")]
        try:
            stored = read(paths, identifier)
        except (OSError, ValueError):
            continue
        if isinstance(stored, dict):
            found.append((identifier, stored))
    return found


def _bare(account_uuid):
    """The id of a login's first organization: its uuid, canonical when it is one."""
    return _canonical_uuid(account_uuid) or account_uuid


def records_of_login(paths, account_uuid):
    """(id, stored organizationUuid) of every record of one login. A record with
    no stored accountUuid counts by its id, as records written before the
    identity was stored did."""
    wanted = str(account_uuid).lower()
    bare = _bare(account_uuid)
    found = []
    for identifier, stored in records(paths):
        login, organization = login_of(stored)
        if login == wanted or (login is None and identifier == bare):
            found.append((identifier, organization))
    return found


def resolve(paths, account_uuid, org_uuid):
    """The account id of a live login, which need not be enrolled. The id is
    opaque, so it is only ever found through the stored records: the record of
    that login and organization; else the login's bare uuid when its record
    stores no organization, or the live one is unknown and the login has one
    record only; else the id a further organization is enrolled under."""
    if not isinstance(account_uuid, str) or not account_uuid:
        return None
    organization = _canonical_uuid(org_uuid)
    bare = _bare(account_uuid)
    found = records_of_login(paths, account_uuid)
    if organization is not None:
        for identifier, stored in found:
            if stored == organization:
                return identifier
    if any(identifier == bare and stored is None for identifier, stored in found):
        return bare
    if organization is None:
        return found[0][0] if len(found) == 1 else bare
    return "%s.%s" % (bare, organization)


def new_id(paths, identity):
    """The id to enrol `identity` under. A login's first organization keeps the
    bare uuid, as every install before organizations were told apart did; each
    further one gets `<accountUuid>.<organizationUuid>`. An organization already
    enrolled keeps its id, and so does a record that stores no organization,
    since nothing says it is another one."""
    bare = account_id(identity.get("accountUuid"))
    organization = _canonical_uuid(identity.get("organizationUuid"))
    found = records_of_login(paths, bare)
    if organization is not None:
        for identifier, stored in found:
            if stored == organization:
                return identifier
    if not os.path.exists(os.path.join(paths.accounts_dir, bare + ".json")):
        return bare
    if any(identifier == bare and stored is None for identifier, stored in found):
        return bare
    if organization is None:
        if len(found) == 1:
            return found[0][0]
        raise RuntimeError(
            "That login names no organization, and grazr holds several of its "
            "organizations, so it cannot tell which one it is"
        )
    return account_id("%s.%s" % (bare, organization))


NAME_SLUG_LENGTH = 24


def default_name(paths, identity, identifier):
    """The name an account is enrolled under when the caller gives none: its
    own when it is enrolled already, the e-mail for a login's first
    organization, `<email>:<org-slug>` for a further one. No whitespace, since
    ACCOUNTS is split on it. None when there is nothing to name it by."""
    try:
        current = read(paths, identifier).get("name")
    except (OSError, ValueError, AttributeError):
        current = None
    if isinstance(current, str) and current:
        return current
    email = identity.get("emailAddress")
    email = "".join(email.split()) if isinstance(email, str) else ""
    bare = account_id(identity.get("accountUuid"))
    if identifier == bare:
        return email or None
    organization = identity.get("organizationName")
    slug = re.sub(r"[^a-z0-9]+", "-", organization.lower() if isinstance(organization, str) else "")
    slug = slug.strip("-")[:NAME_SLUG_LENGTH].strip("-")
    slug = slug or (_canonical_uuid(identity.get("organizationUuid")) or "org")[:8]
    base = "%s:%s" % (email or bare[:8], slug)
    taken = {entry.name for entry in load(paths, []) if entry.id != identifier}
    name, suffix = base, 2
    while name in taken:
        name, suffix = "%s-%d" % (base, suffix), suffix + 1
    return name


def has_parked_credential(store, identifier):
    """An account whose parked credential went missing cannot be rotated to,
    and the only way to find out is to look."""
    return store.read_parked(identifier) is not None


def plan_of(oauth):
    """The plan an OAuth profile names, or None without an organizationType.
    `oauth` is the `oauthAccount` object from Claude's config or a stored record."""
    if not isinstance(oauth, dict) or not oauth.get("organizationType"):
        return None
    return {
        "type": oauth["organizationType"],
        "tier": oauth.get("organizationRateLimitTier"),
        "seat_tier": oauth.get("userRateLimitTier"),
    }


def load(paths, names):
    """Enrolled accounts in the configured preference order. An unknown name is
    skipped rather than fatal, so a typo cannot stop rotation entirely."""
    by_name = {}
    for filename in sorted(os.listdir(paths.accounts_dir)):
        if filename.startswith(".") or not filename.endswith(".json"):
            continue
        identifier = filename[: -len(".json")]
        try:
            stored = read(paths, identifier)
        except (OSError, ValueError):
            continue
        # Valid JSON is not yet a valid account.
        if not isinstance(stored, dict) or not isinstance(stored.get("name", identifier), str):
            continue
        by_name[stored.get("name", identifier)] = core.Account(
            id=identifier,
            name=stored.get("name", identifier),
            snapshot=snapshot_from_json(stored.get("snapshot")),
            plan=plan_of(stored.get("oauthAccount")),
        )
    if not names:
        return [by_name[name] for name in sorted(by_name)]
    return [by_name[name] for name in names if name in by_name]


def read(paths, identifier):
    with open(os.path.join(paths.accounts_dir, identifier + ".json")) as handle:
        return json.load(handle)


def write(paths, identifier, account):
    """Metadata only -- never a secret. The credential lives in the store."""
    atomic.write(
        os.path.join(paths.accounts_dir, identifier + ".json"),
        json.dumps(account, indent=2),
    )


def record_snapshot(paths, identifier, snapshot):
    """Best-effort: the swap has already happened, so an account we never
    enrolled is nothing to record against rather than a reason to fail."""
    try:
        account = read(paths, identifier)
    except (OSError, ValueError):
        return
    account["snapshot"] = snapshot_to_json(snapshot)
    write(paths, identifier, account)


def mark_first_reading_due(paths, identifier, due=True):
    """Set by the swap that installs this account, cleared by the first
    reading that shows a turn on it."""
    try:
        account = read(paths, identifier)
    except (OSError, ValueError):
        return
    if due:
        account["first_reading_due"] = True
    else:
        account.pop("first_reading_due", None)
    write(paths, identifier, account)


def first_reading_due(paths, identifier):
    try:
        return read(paths, identifier).get("first_reading_due") is True
    except (OSError, ValueError, AttributeError):
        return False


def updated_at(paths, identifier):
    """Unix seconds of the account's last recorded reading: the mtime of its
    state file. None when the file cannot be read."""
    try:
        return os.path.getmtime(os.path.join(paths.accounts_dir, identifier + ".json"))
    except OSError:
        return None


def snapshot_to_json(snapshot):
    if not isinstance(snapshot, list):
        return None
    return [
        {
            "kind": entry.kind,
            "scope": entry.scope,
            "group": entry.group,
            "remaining": entry.remaining,
            "resets_at": entry.resets_at.isoformat() if entry.resets_at else None,
        }
        for entry in snapshot
    ]


def snapshot_from_json(stored):
    """Anything unrecognised reads as never parked. A hand-edited account file
    must not take rotation out for every account."""
    if stored is None:
        return None
    try:
        limits = [
            core.Limit(
                kind=entry["kind"],
                scope=entry["scope"],
                group=entry["group"],
                remaining=entry["remaining"],
                resets_at=parse_time(entry["resets_at"]),
            )
            for entry in stored
        ]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    # The decision compares these, and a string or a bool would compare wrong
    # rather than raise.
    if all(
        isinstance(entry.kind, str)
        and isinstance(entry.group, str)
        and isinstance(entry.remaining, (int, float))
        and not isinstance(entry.remaining, bool)
        for entry in limits
    ):
        return limits
    return None
