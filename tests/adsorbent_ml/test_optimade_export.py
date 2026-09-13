"""Offline tests for the OPTIMADE exporter (ACQUISITION §5).

All HTTP is faked: discovery (info-vs-links protocol), pagination,
failure modes and the JSONL/manifest cache contract.
"""

import json

import pytest

from optimade_export import (export_structures, query_structures,
                             resolve_queryable_bases)


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _fake_get_factory():
    pages = {
        "https://idx.example/links": _Resp({"data": [
            {"attributes": {"name": "ProvA", "base_url": "https://a.example/db"}},
            {"attributes": {"name": "IdxB", "base_url": "https://b.example/idx"}},
        ]}),
        "https://a.example/db/info": _Resp({"data": {"type": "other"}}),
        "https://a.example/db/links": _Resp({"data": []}),
        "https://a.example/db/v1/info": _Resp({"data": {
            "type": "info",
            "attributes": {"available_endpoints": ["info", "structures"]}}}),
        "https://b.example/idx/info": _Resp(status=404),
        "https://b.example/idx/links": _Resp({"data": [
            {"attributes": {"name": "ProvC", "base_url": "https://c.example/v1"}},
        ]}),
        "https://c.example/v1/info": _Resp({"data": {
            "type": "info",
            "attributes": {"available_endpoints": ["info", "structures"]}}}),
    }

    def get(url, **kw):
        if url.startswith("https://a.example/db/v1/structures"):
            if "page_cursor=next" in url:
                return _Resp({"data": [{"id": "e3"}], "links": {}})
            return _Resp({"data": [{"id": "e1"}, {"id": "e2"}],
                          "links": {"next": "https://a.example/db/v1/structures?page_cursor=next"}})
        if url in pages:
            return pages[url]
        return _Resp(status=404)

    return get


def test_discovery_resolves_queryable_and_recurses_index():
    bases = resolve_queryable_bases("https://idx.example/links",
                                    get=_fake_get_factory())
    by_name = {b["provider"]: b["base_url"] for b in bases}
    assert by_name["ProvA"] == "https://a.example/db/v1"
    assert by_name["ProvC"] == "https://c.example/v1"


def test_index_info_doc_is_not_mistaken_for_database():
    """Regression (seen live on the COD static index): an index meta-database
    answers /info with type 'info' but no 'structures' endpoint — the
    resolver must recurse into its /links, not list it as queryable."""

    def get(url, **kw):
        if url == "https://idx.example/links":
            return _Resp({"data": [
                {"attributes": {"name": "Idx",
                                "base_url": "https://idx.example/sub"}},
            ]})
        if url == "https://idx.example/sub/info":
            return _Resp({"data": {"type": "info", "attributes": {
                "available_endpoints": ["info", "links"]}}})
        if url == "https://idx.example/sub/links":
            return _Resp({"data": [
                {"attributes": {"name": "Real",
                                "base_url": "https://real.example/v1"}},
            ]})
        if url == "https://real.example/v1/info":
            return _Resp({"data": {"type": "info", "attributes": {
                "available_endpoints": ["info", "structures"]}}})
        return _Resp(status=404)

    bases = resolve_queryable_bases("https://idx.example/links", get=get)
    assert [(b["provider"], b["base_url"]) for b in bases] == [
        ("Real", "https://real.example/v1")]


def test_query_follows_pagination_and_caps():
    get = _fake_get_factory()
    all_ids = [e["id"] for e in query_structures("https://a.example/db/v1", "nelements<5",
                                                 max_entries=10, get=get)]
    assert all_ids == ["e1", "e2", "e3"]
    capped = [e["id"] for e in query_structures("https://a.example/db/v1", "nelements<5",
                                                max_entries=2, get=get)]
    assert capped == ["e1", "e2"]


def test_http_error_and_bad_document_fail_loudly():
    with pytest.raises(RuntimeError, match="HTTP 404"):
        list(query_structures("https://missing.example/v1", "nelements<5",
                              get=_fake_get_factory()))
    bad = lambda url, **kw: _Resp({"not": "jsonapi"})  # noqa: E731
    with pytest.raises(ValueError, match="not a JSON:API document"):
        list(query_structures("https://a.example/db/v1", "nelements<5", get=bad))


def test_export_writes_jsonl_and_manifest(tmp_path):
    manifest = export_structures("provA", "nelements<5", max_entries=10,
                                 out_dir=tmp_path, index_url="https://idx.example/links",
                                 get=_fake_get_factory())
    assert manifest["n_entries"] == 3 and manifest["capped"] is False
    assert manifest["filter"] == "nelements<5"
    lines = (tmp_path / "structures.jsonl").read_text().strip().split("\n")
    assert [json.loads(l)["id"] for l in lines] == ["e1", "e2", "e3"]
    assert json.loads((tmp_path / "manifest.json").read_text())["n_entries"] == 3


def test_export_no_provider_match_lists_resolved():
    with pytest.raises(ValueError, match="ProvA"):
        export_structures("no-such-provider", index_url="https://idx.example/links",
                          get=_fake_get_factory())
