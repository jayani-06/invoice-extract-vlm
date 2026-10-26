"""Who is acting, and what they are allowed to do.

Before this, the service took the reviewer's name as a plain string in the
request body. An override of a FAIL was therefore *recorded* against a name,
never *verified* against an identity: anyone who could reach the port could
make a blocked invoice payable under any name they liked, including someone
else's.

Two things are fixed, and the second matters more than the first:

1. **The identity comes from the credential, not the body.** `reviewer` is no
   longer something a caller asserts about itself. A request carries a token;
   the token names a principal; the audit trail records that principal. A body
   that disagrees is rejected rather than silently preferred, so a stale client
   fails loudly instead of writing a wrong name into the log.

2. **Submitting and approving are separate rights.** The whole point of a
   review queue is that something a machine would not pass gets a second,
   independent pair of eyes. If the account that uploads an invoice can also
   clear its own FAIL, there is no second pair of eyes and the queue is
   decoration. `SUBMIT` and `OVERRIDE` are distinct scopes and a principal can
   hold either, both, or neither.

**What this is not.** Tokens are shared secrets compared in constant time, read
from the environment. There is no password hashing, no rotation, no expiry, no
transport security, and no identity provider — this authenticates a *caller*,
not a *person*, and two people sharing a token are indistinguishable. It is
honest access control for a prototype on a trusted network and it is not a
substitute for SSO. `docs/phase5_service.md` says so in the same words.

**Unconfigured means open, loudly.** With no tokens set the service runs
exactly as it did before and every request is the `anonymous` principal holding
every scope. That keeps local development and the test suite working, but
`create_app` warns on startup and `/health` reports `auth: "disabled"`, because
a security control that is silently absent is worse than one that is absent.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

#: Environment variable holding the token table, as
#: `name:scope[+scope]:token` entries separated by commas or newlines.
#:
#:     INVOICE_SERVICE_TOKENS="ingest:submit:s3cr3t,ravi:review+override:h3nt3r"
#:
#: Read from the environment rather than a file so a token never lands in the
#: repository by accident.
TOKEN_ENV = "INVOICE_SERVICE_TOKENS"


class Scope(str, Enum):
    """What a principal may do."""

    SUBMIT = "submit"
    """Upload documents for processing."""

    REVIEW = "review"
    """Clear abstentions and accept documents the pipeline did not block."""

    OVERRIDE = "override"
    """Make a FAILED document payable. Deliberately separate from SUBMIT: if
    the uploader can clear their own block, the review queue is decoration."""

    READ = "read"
    """Read the queue, documents and reference data."""


ALL_SCOPES = frozenset(Scope)


@dataclass(frozen=True)
class Principal:
    """Who the request is from."""

    name: str
    scopes: frozenset[Scope] = field(default_factory=frozenset)

    @property
    def anonymous(self) -> bool:
        return self.name == "anonymous"

    def may(self, scope: Scope) -> bool:
        return scope in self.scopes


ANONYMOUS = Principal(name="anonymous", scopes=ALL_SCOPES)


def parse_tokens(raw: str | None) -> dict[str, Principal]:
    """Build the token table from its environment form.

    Malformed entries raise rather than being skipped. A typo that silently
    dropped a principal would leave the service running with fewer credentials
    than its operator believes it has, which is the kind of failure that is
    only discovered when someone cannot do their job -- or worse, is not
    discovered at all because the remaining token is over-privileged.
    """
    table: dict[str, Principal] = {}
    if not raw or not raw.strip():
        return table

    for entry in raw.replace("\n", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"token entry {entry.split(':')[0]!r} is malformed; "
                "expected name:scope[+scope]:token"
            )
        name, scope_text, token = (p.strip() for p in parts)
        if not name or not token:
            raise ValueError(f"token entry {name!r} has an empty name or token")
        scopes = set()
        for word in scope_text.split("+"):
            word = word.strip().lower()
            if not word:
                continue
            try:
                scopes.add(Scope(word))
            except ValueError:
                raise ValueError(
                    f"token entry {name!r} has unknown scope {word!r}; "
                    f"known scopes are {sorted(s.value for s in Scope)}"
                ) from None
        # READ is implied by every other scope: a reviewer who cannot see the
        # queue cannot review, and requiring it to be spelled out is a trap.
        if scopes:
            scopes.add(Scope.READ)
        table[token] = Principal(name=name, scopes=frozenset(scopes))
    return table


class Authenticator:
    """Resolves a bearer token to a principal."""

    def __init__(self, tokens: dict[str, Principal] | None = None) -> None:
        self.tokens = dict(tokens or {})

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Authenticator:
        source = env if env is not None else os.environ
        return cls(parse_tokens(source.get(TOKEN_ENV)))

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def principal_for(self, header: str | None) -> Principal:
        """The principal a request belongs to.

        With no tokens configured every caller is `anonymous` holding every
        scope, which is what the service did before this existed.
        """
        if not self.enabled:
            return ANONYMOUS

        token = (header or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "this endpoint needs a bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Constant-time comparison against every known token. A dict lookup
        # would leak which prefixes are close to a real one through timing,
        # and the table is small enough that scanning it costs nothing.
        for known, principal in self.tokens.items():
            if hmac.compare_digest(token, known):
                return principal

        # Deliberately the same message as the missing-token case, and the
        # token itself is never echoed or logged.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "unrecognised token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def require(self, principal: Principal, scope: Scope) -> None:
        if not principal.may(scope):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"{principal.name!r} may not {scope.value}",
            )


def authorization_header(authorization: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency returning the raw Authorization header."""
    return authorization


def describe(auth: Authenticator) -> dict[str, object]:
    """What `/health` reports. Never the tokens themselves."""
    if not auth.enabled:
        return {"auth": "disabled", "principals": []}
    return {
        "auth": "token",
        "principals": sorted(
            {p.name: sorted(s.value for s in p.scopes) for p in auth.tokens.values()}.items()
        ),
    }
