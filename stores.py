"""Credential stores: where the live and parked secrets are kept.

Callers never learn which store they are talking to; how items are named and
moved is each store's own business.
"""

import hashlib
import os
import platform
import re
import subprocess
import unicodedata

import atomic

IS_MAC = platform.system() == "Darwin"

SERVICE = "Claude Code-credentials"

# Absolute, so a `security` planted earlier on PATH cannot be handed the
# credential. Apple's own binary is also the one the keychain binds its access
# grant to, which is why grazr shells out here rather than linking a framework:
# a grant against a rebuilt binary of our own would die at the next build.
SECURITY_BIN = "/usr/bin/security"

# A swap makes three keychain calls. All three at full timeout must still fit
# inside the lock stale ages claude.py mirrors.
SECURITY_TIMEOUT_SECONDS = 15

# errSecItemNotFound. Any other non-zero exit means the keychain itself was
# unhappy, which is not the same as the item being absent.
ITEM_NOT_FOUND = 44


class FileStore:
    """Linux has no keychain: Claude keeps the live credential in a file only
    the owner can read, and grazr parks copies with the same protection."""

    def __init__(self, live_path, parked_dir):
        self.live_path = live_path
        self.parked_dir = parked_dir

    def read_live(self):
        return self._read(self.live_path)

    def write_live(self, blob):
        self._write(self.live_path, blob)

    def read_parked(self, account_id):
        return self._read(self._parked_path(account_id))

    def write_parked(self, account_id, blob):
        os.makedirs(self.parked_dir, mode=0o700, exist_ok=True)
        self._write(self._parked_path(account_id), blob)

    def read_isolated(self, config_dir):
        return self._read(self._isolated_path(config_dir))

    def discard_isolated(self, config_dir):
        """Never raises: the caller runs in a finally."""
        try:
            os.unlink(self._isolated_path(config_dir))
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return True

    def _parked_path(self, account_id):
        return os.path.join(self.parked_dir, account_id + ".json")

    def _isolated_path(self, config_dir):
        return os.path.join(config_dir, ".credentials.json")

    def _read(self, path):
        try:
            with open(path) as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def _write(self, path, blob):
        """Claude re-reads this file on every request and must never see it
        half-written."""
        atomic.write(path, blob)


def default_store(isolated, config_dir, state_dir, keychain_account):
    """The store the running platform's Claude actually reads."""
    if IS_MAC:
        return KeychainStore(service_name(isolated), keychain_account)
    return FileStore(
        os.path.join(config_dir, ".credentials.json"),
        os.path.join(state_dir, "credentials"),
    )


def service_name(config_dir=None):
    """Claude's own scheme: one item per config directory. NFC-normalise before
    hashing, because the filesystem can return an umlaut as two characters and
    the same directory must always hash the same."""
    if config_dir is None:
        return SERVICE
    normalized = unicodedata.normalize("NFC", config_dir)
    return "%s-%s" % (SERVICE, hashlib.sha256(normalized.encode()).hexdigest()[:8])


def _scrubbed(stderr):
    """security echoes the tail of the blob in its error and the message
    reaches the plugin log. Scrub before trimming, or the cut can leave a hex
    fragment too short to match."""
    return re.sub(r"[0-9a-f]{8,}", "<hex>", stderr.strip())[:200]


class KeychainStore:
    def __init__(self, service, keychain_account, spawn=subprocess.run):
        self.service = service
        self.keychain_account = keychain_account
        self._spawn = spawn

    def read_live(self):
        return self._read(self.service)

    def write_live(self, blob):
        self._install(self.service, blob)

    def read_parked(self, account_id):
        return self._read(self._parked_service(account_id))

    def write_parked(self, account_id, blob):
        self._install(self._parked_service(account_id), blob)

    def read_isolated(self, config_dir):
        return self._read(service_name(config_dir))

    def discard_isolated(self, config_dir):
        """Never raises: the caller runs in a finally, and masking the failure
        it is cleaning up after would hide the real problem."""
        try:
            completed = self._spawn(
                [
                    SECURITY_BIN,
                    "delete-generic-password",
                    "-s",
                    service_name(config_dir),
                    "-a",
                    self.keychain_account,
                ],
                capture_output=True,
                timeout=SECURITY_TIMEOUT_SECONDS,
            )
            return getattr(completed, "returncode", 0) in (0, ITEM_NOT_FOUND)
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _parked_service(self, account_id):
        return "grazr-%s" % account_id

    def _read(self, service):
        """Service and account are not secrets, so argv is fine here; the blob
        comes back on stdout."""
        try:
            completed = self._spawn(
                [SECURITY_BIN, "find-generic-password", "-s", service, "-a", self.keychain_account, "-w"],
                capture_output=True,
                text=True,
                timeout=SECURITY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("The keychain did not answer within %ds" % SECURITY_TIMEOUT_SECONDS)
        if completed.returncode == ITEM_NOT_FOUND:
            return None
        if completed.returncode != 0:
            raise RuntimeError(
                "The keychain refused to read the credential: %s"
                % (_scrubbed(completed.stderr) or completed.returncode)
            )
        stored = completed.stdout.strip()
        try:
            return bytes.fromhex(stored).decode()
        except (ValueError, UnicodeDecodeError):
            return stored

    def _install(self, service, blob):
        """The secret goes as a hex argument, the way Claude Code writes the
        item itself. The tool's interactive line truncates past 4095 bytes and
        destroys the item, so a login with many MCP servers never fit through
        it. An argument has no such limit, and the secret shows in the process
        list only for the moment the tool runs, as it does on every login and
        token refresh Claude makes."""
        try:
            completed = self._spawn(
                [
                    SECURITY_BIN, "add-generic-password", "-U",
                    "-s", service, "-a", self.keychain_account, "-X", blob.encode().hex(),
                ],
                capture_output=True,
                text=True,
                timeout=SECURITY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("The keychain did not answer within %ds" % SECURITY_TIMEOUT_SECONDS)
        if completed.returncode != 0:
            raise RuntimeError(
                "The security tool refused the command: %s" % _scrubbed(completed.stderr)
            )
        return completed.stdout
