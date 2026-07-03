from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_compose_wires_crawler_caddy_and_optional_local_llm():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "crawler:" in compose
    assert "FRONTIER_STATE_PATH=/data/frontier.json" in compose
    assert "crawler-data:/data" in compose
    assert "caddy:" in compose
    assert "Caddyfile" in compose
    assert 'profiles: ["local-llm"]' in compose


def test_run_crawler_script_uses_persistent_frontier_and_indexing():
    script = (ROOT / "scripts" / "run_crawler.py").read_text()
    assert "SeedManager.load" in script
    assert "seeds.save(FRONTIER_STATE_PATH)" in script
    assert "run_crawl_cycle" in script
    assert "index_page" in script


def test_caddyfile_proxies_to_app():
    caddyfile = (ROOT / "Caddyfile").read_text()
    assert "reverse_proxy app:8000" in caddyfile


def test_no_manual_content_blocklist_remains_in_config_or_scripts():
    # Dark web is the only content exclusion; no BLOCKED_DOMAINS knob may exist.
    for rel in (
        ".env.example",
        "docker-compose.yml",
        "scripts/backfill_common_crawl.py",
        "scripts/run_crawler.py",
    ):
        assert "BLOCKED_DOMAINS" not in (ROOT / rel).read_text(), rel
