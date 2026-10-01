from collections import namedtuple

Limit = namedtuple("Limit", "kind scope group remaining resets_at")
Account = namedtuple("Account", "id name snapshot")

# An account with this little or less left in any open window is spent. The
# thresholds are a margin for leaving early, and once the live account is spent
# any account with more left beats staying on it, margin or not.
SPENT_AT = 2


def merged(previous, current):
    """A window already on record keeps the lowest headroom it has shown. Idle
    panes repeat old figures, and a repeat must not put headroom back. A newer
    window is taken as it comes. An older one comes from a pane that has not
    caught up, and the window on record stays."""
    if not isinstance(previous, list):
        return current
    # Keyed by scope too: a per-model weekly limit shares its reset with the
    # all-models one, and must never lend it its lower headroom.
    lowest = {(entry.group, entry.scope, entry.resets_at): entry.remaining for entry in previous}
    newest = {
        (entry.kind, entry.group, entry.scope): entry
        for entry in sorted((entry for entry in previous if entry.resets_at), key=lambda entry: entry.resets_at)
    }
    readings = []
    for entry in current:
        recorded = newest.get((entry.kind, entry.group, entry.scope))
        if recorded and entry.resets_at and entry.resets_at < recorded.resets_at:
            readings.append(recorded)
            continue
        readings.append(
            entry._replace(
                remaining=min(
                    entry.remaining, lowest.get((entry.group, entry.scope, entry.resets_at), entry.remaining)
                )
            )
        )
    return readings


def needs_rotation(limits, now, thresholds):
    return not _has_headroom(limits, now, thresholds)


def decide(limits, active, accounts, now, thresholds):
    if not needs_rotation(limits, now, thresholds):
        return "stay"
    chosen = next_account(active, accounts, now, thresholds)
    if chosen:
        return "rotate", chosen
    if is_spent(limits, now, thresholds):
        chosen = last_resort(active, accounts, now, thresholds)
        if chosen:
            return "rotate", chosen
    # "Every account is spent" would be a claim about accounts that do not exist.
    return "exhausted" if any(entry.id != active for entry in accounts) else "unenrolled"


def next_account(active, accounts, now, thresholds):
    """The first account in preference order fit to take over, or None. Says
    nothing about whether the active one needs replacing."""
    for candidate in accounts:
        if candidate.id == active:
            continue
        if candidate.snapshot is None or _has_headroom(candidate.snapshot, now, thresholds):
            return candidate.id
    return None


def is_spent(limits, now, thresholds):
    return _least_left(limits, now, thresholds) <= SPENT_AT


def last_resort(active, accounts, now, thresholds):
    """For a spent live account when nobody clears the thresholds: the account
    with the most left in its tightest window, or None if every one is spent
    too. Ties go to preference order."""
    chosen, most = None, SPENT_AT
    for candidate in accounts:
        if candidate.id == active or candidate.snapshot is None:
            continue
        left = _least_left(candidate.snapshot, now, thresholds)
        if left > most:
            chosen, most = candidate.id, left
    return chosen


def _least_left(limits, now, thresholds):
    """The lowest headroom among open windows grazr watches, 100 with none."""
    return min(
        (
            limit.remaining
            for limit in (limits if isinstance(limits, list) else [])
            if limit.group in thresholds and (limit.resets_at is None or limit.resets_at > now)
        ),
        default=100,
    )


def _has_headroom(limits, now, thresholds):
    return all(
        limit.remaining >= thresholds[limit.group]
        for limit in limits
        if limit.group in thresholds and (limit.resets_at is None or limit.resets_at > now)
    )
