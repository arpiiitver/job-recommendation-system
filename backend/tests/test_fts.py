"""Full-text job search (Postgres FTS): stemming, phrases, sorting, filters.

Runs in-process through the API (no live server needed) against the seeded
golden corpus (scripts/seed_demo.py). Live Adzuna results are blocked by
conftest, so only platform postings are searched.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _search(**params):
    resp = client.get("/api/v1/jobs", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_fts_stemming_matches_singular_title():
    # "engineers" must match postings titled "...Engineer"
    res = _search(q="engineers")
    assert res["total"] > 0
    assert any("engineer" in item["title"].lower() for item in res["items"])


def test_fts_phrase_search():
    res = _search(q='"Machine Learning"')
    assert res["total"] > 0


def test_fts_sort_by_salary_is_descending():
    items = _search(sort_by="salary")["items"]
    salaries = [i["salary_max"] or i["salary_min"] for i in items if (i["salary_max"] or i["salary_min"])]
    assert salaries == sorted(salaries, reverse=True)


def test_fts_multi_filter():
    res = _search(q="python", is_remote="true", min_salary="100000")
    assert res["total"] > 0
    for item in res["items"]:
        assert item["is_remote"] is True
