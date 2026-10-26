"""Who may do what, and whose name ends up in the audit trail.

The service took the reviewer's name as a plain string in the request body, so
an override of a FAIL was *recorded* against a name and never *verified*
against an identity. Anyone who could reach the port could make a blocked
invoice payable under any name, including someone else's.

Two properties matter more than the mechanics, and both fail silently rather
than loudly if they regress:

- **The identity comes from the credential.** If the body can still name the
  reviewer, the audit trail is a suggestion.
- **Submitting and approving are separate.** The point of a review queue is a
  second, independent pair of eyes. If the account that uploads an invoice can
  clear its own FAIL, there is no second pair of eyes and the queue is
  decoration.

The third property is about honesty rather than security: with no tokens
configured the service behaves exactly as before, and says so, because a
control that is silently absent is worse than one that is absent.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from invoice_extract.service.auth import (
    ANONYMOUS,
    Authenticator,
    Principal,
    Scope,
    describe,
    parse_tokens,
)


class TestParsingTheTokenTable:
    def test_a_principal_gets_the_scopes_it_was_given(self):
        table = parse_tokens("ravi:review+override:secret")
        principal = table["secret"]
        assert principal.name == "ravi"
        assert principal.may(Scope.REVIEW)
        assert principal.may(Scope.OVERRIDE)
        assert not principal.may(Scope.SUBMIT)

    def test_read_is_implied_by_any_other_scope(self):
        """A reviewer who cannot see the queue cannot review. Requiring READ to
        be spelled out is a trap that only shows up in production."""
        assert parse_tokens("ravi:review:s")["s"].may(Scope.READ)

    def test_several_principals(self):
        table = parse_tokens("ingest:submit:a,ravi:review+override:b")
        assert {p.name for p in table.values()} == {"ingest", "ravi"}

    def test_newlines_are_accepted_as_separators(self):
        assert len(parse_tokens("ingest:submit:a\nravi:review:b")) == 2

    def test_empty_means_no_tokens(self):
        assert parse_tokens(None) == {}
        assert parse_tokens("   ") == {}

    @pytest.mark.parametrize(
        "raw",
        ["ravi:review", "ravi:review:", ":review:tok", "ravi:nonsense:tok"],
    )
    def test_a_malformed_entry_raises_rather_than_being_skipped(self, raw):
        """Skipping a typo leaves the service running with fewer credentials
        than its operator believes, which is discovered when someone cannot do
        their job -- or never, because the remaining token is over-privileged."""
        with pytest.raises(ValueError):
            parse_tokens(raw)

    def test_the_error_never_contains_the_token(self):
        with pytest.raises(ValueError) as excinfo:
            parse_tokens("ravi:nonsense:supersecret")
        assert "supersecret" not in str(excinfo.value)


class TestResolvingACaller:
    @pytest.fixture
    def auth(self) -> Authenticator:
        return Authenticator(parse_tokens("ingest:submit:aaa,ravi:review+override:bbb"))

    def test_a_valid_token_resolves(self, auth):
        assert auth.principal_for("Bearer aaa").name == "ingest"

    def test_the_bearer_prefix_is_optional(self, auth):
        assert auth.principal_for("aaa").name == "ingest"

    def test_a_missing_token_is_rejected(self, auth):
        with pytest.raises(HTTPException) as excinfo:
            auth.principal_for(None)
        assert excinfo.value.status_code == 401

    def test_an_unknown_token_is_rejected(self, auth):
        with pytest.raises(HTTPException) as excinfo:
            auth.principal_for("Bearer nope")
        assert excinfo.value.status_code == 401

    def test_a_rejection_never_echoes_the_token(self, auth):
        """A token in an error body ends up in a log, a browser history or a
        screenshot."""
        with pytest.raises(HTTPException) as excinfo:
            auth.principal_for("Bearer hunter2")
        assert "hunter2" not in str(excinfo.value.detail)

    def test_the_scope_check_refuses_what_the_principal_lacks(self, auth):
        ingest = auth.principal_for("aaa")
        auth.require(ingest, Scope.SUBMIT)  # allowed
        with pytest.raises(HTTPException) as excinfo:
            auth.require(ingest, Scope.OVERRIDE)
        assert excinfo.value.status_code == 403


class TestUnconfiguredMeansOpenButLoud:
    def test_every_caller_is_anonymous_with_every_scope(self):
        auth = Authenticator({})
        assert not auth.enabled
        who = auth.principal_for(None)
        assert who == ANONYMOUS
        for scope in Scope:
            assert who.may(scope)

    def test_health_says_so(self):
        """Silently absent is worse than absent."""
        assert describe(Authenticator({}))["auth"] == "disabled"

    def test_health_never_reports_a_token(self):
        auth = Authenticator(parse_tokens("ravi:review:supersecret"))
        assert "supersecret" not in repr(describe(auth))
        assert describe(auth)["auth"] == "token"


class TestSeparationOfDuties:
    def test_submitting_does_not_confer_overriding(self):
        """The property the whole thing exists for. An uploader who can clear
        its own FAIL turns the review queue into decoration."""
        ingest = Principal("ingest", frozenset({Scope.SUBMIT, Scope.READ}))
        assert ingest.may(Scope.SUBMIT)
        assert not ingest.may(Scope.OVERRIDE)

    def test_reviewing_does_not_confer_overriding(self):
        """Clearing an abstention is routine; making a blocked invoice payable
        is not, so they are separate rights."""
        reviewer = Principal("ravi", frozenset({Scope.REVIEW, Scope.READ}))
        assert reviewer.may(Scope.REVIEW)
        assert not reviewer.may(Scope.OVERRIDE)


class TestTheApiEnforcesIt:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from invoice_extract.service.api import create_app

        auth = Authenticator(parse_tokens("ingest:submit:aaa,ravi:review+override:bbb"))
        # No ReviewService: these requests are refused before any handler body
        # runs, so none of them needs the pipeline.
        return TestClient(create_app(object(), auth=auth), raise_server_exceptions=False)

    def test_an_unauthenticated_upload_is_refused(self, client):
        resp = client.post("/documents", files={"file": ("x.png", b"x", "image/png")})
        assert resp.status_code == 401

    def test_a_reviewer_token_may_not_upload(self, client):
        resp = client.post(
            "/documents",
            files={"file": ("x.png", b"x", "image/png")},
            headers={"Authorization": "Bearer bbb"},
        )
        assert resp.status_code == 403
        assert "ravi" in resp.json()["detail"]

    def test_an_ingest_token_may_not_decide(self, client):
        resp = client.post(
            "/review/lines/1",
            json={"reviewer": "ingest", "action": "accept"},
            headers={"Authorization": "Bearer aaa"},
        )
        assert resp.status_code == 403

    def test_health_is_open(self, client):
        """Something has to be reachable without a credential, or a load
        balancer cannot tell whether the service is alive."""
        assert client.get("/health").status_code in (200, 500)
