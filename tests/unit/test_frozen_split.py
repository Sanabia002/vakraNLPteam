"""The tune sample must contain no frozen-test-domain or seen questions."""
import json

import pytest

from frozen_split import FROZEN_TEST, SEEN_UUIDS, assert_no_frozen, uuid_domains

DOMAINS = uuid_domains()


def test_tune_sample_has_no_frozen_or_seen_uuids():
    sample = json.load(open("results/tune_sample_uuids.json"))
    assert len(sample) == len(set(sample)) == 68
    assert_no_frozen(sample, DOMAINS)


def test_seen_uuid_is_the_bike_share_smoke_question():
    assert {DOMAINS[u] for u in SEEN_UUIDS} == {"bike_share_1"}


def test_assertion_fires_on_frozen_domain():
    frozen_uuid = next(u for u, d in DOMAINS.items() if d in FROZEN_TEST and u not in SEEN_UUIDS)
    ok = json.load(open("results/tune_sample_uuids.json"))
    with pytest.raises(AssertionError, match="frozen test domains"):
        assert_no_frozen(ok + [frozen_uuid], DOMAINS)


def test_assertion_fires_on_seen_uuid_even_if_domain_unfrozen():
    u = next(iter(SEEN_UUIDS))
    with pytest.raises(AssertionError, match="seen uuids"):
        assert_no_frozen([u], {u: "authors"})


def test_assertion_fires_on_unknown_uuid():
    with pytest.raises(AssertionError, match="no known domain"):
        assert_no_frozen(["not-a-uuid"], DOMAINS)
