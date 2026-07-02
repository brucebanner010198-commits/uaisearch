from uaisearch.crawler import SeedManager


def test_seed_manager_returns_seeds_in_order():
    manager = SeedManager(["https://a.example/1", "https://a.example/2"])
    assert manager.next_url() == "https://a.example/1"
    assert manager.next_url() == "https://a.example/2"
    assert manager.next_url() is None


def test_seed_manager_deduplicates_discovered_urls():
    manager = SeedManager(["https://a.example/1"])
    manager.add_discovered("https://a.example/1")  # already seen, ignored
    manager.add_discovered("https://a.example/2")
    assert len(manager) == 2


def test_seed_manager_save_and_load_round_trips_state(tmp_path):
    path = str(tmp_path / "frontier.json")
    manager = SeedManager(["https://a.example/1"])
    manager.add_discovered("https://a.example/2")
    manager.next_url()  # consume one, so queue and seen differ
    manager.save(path)

    loaded = SeedManager.load(path, default_seeds=["https://unused.example/"])
    assert loaded.next_url() == "https://a.example/2"
    assert len(loaded) == 0

    before = len(loaded)
    loaded.add_discovered("https://a.example/1")  # consumed before save, but still seen
    assert len(loaded) == before  # seen-set survived the round trip
    loaded.add_discovered("https://brand-new.example/")
    assert len(loaded) == before + 1


def test_seed_manager_load_falls_back_to_default_seeds_when_missing(tmp_path):
    path = str(tmp_path / "does-not-exist.json")
    loaded = SeedManager.load(path, default_seeds=["https://a.example/1"])
    assert loaded.next_url() == "https://a.example/1"


def test_load_falls_back_to_default_seeds_on_corrupt_file(tmp_path):
    state = tmp_path / "frontier.json"
    state.write_text("not valid json {{{")
    manager = SeedManager.load(str(state), default_seeds=["https://fallback.example/"])
    assert manager.next_url() == "https://fallback.example/"


import time

import httpx

from uaisearch.crawler import Frontier


def _frontier_with_robots(robots_txt: str) -> Frontier:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=robots_txt)

    return Frontier(http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_can_fetch_respects_disallow():
    frontier = _frontier_with_robots("User-agent: *\nDisallow: /private/\n")
    assert frontier.can_fetch("https://example.com/public/page") is True
    assert frontier.can_fetch("https://example.com/private/page") is False


def test_wait_if_needed_enforces_crawl_delay():
    # urllib.robotparser silently drops non-integer delays and Frontier.crawl_delay
    # floors at 1.0s, so a float delay passes via fallback without testing parsing.
    # Only an integer delay above the floor proves robots.txt was honored.
    frontier = _frontier_with_robots("User-agent: *\nCrawl-delay: 2\n")
    frontier.wait_if_needed("example.com")
    start = time.monotonic()
    frontier.wait_if_needed("example.com")
    assert time.monotonic() - start >= 2


from uaisearch.crawler import is_dark_web


def test_is_dark_web_rejects_onion_and_i2p_hosts():
    assert is_dark_web("http://exampleonionaddr.onion/page") is True
    assert is_dark_web("http://example.i2p/page") is True
    # explicit ports must not bypass the exclusion (netloc includes :port; hostname does not)
    assert is_dark_web("http://example.onion:8080/page") is True
    assert is_dark_web("http://example.i2p:7657/console") is True
    assert is_dark_web("https://ok.example:8443/") is False


def test_is_dark_web_allows_ordinary_hosts():
    assert is_dark_web("https://example.com/page") is False


from uaisearch.crawler import fetch


async def test_fetch_returns_response_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>hi</body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await fetch("https://example.com/", client)
    assert "hi" in text


async def test_fetch_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        try:
            await fetch("https://example.com/missing", client)
            assert False, "expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass


from uaisearch.crawler import extract_clean_text, strip_ad_elements


def test_strip_ad_elements_removes_ad_classed_tags():
    html = '<div class="sponsored-banner">ad</div><div class="content">real content</div>'
    soup, ad_ratio = strip_ad_elements(html)
    assert soup.find("div", class_="sponsored-banner") is None
    assert soup.find("div", class_="content") is not None
    assert ad_ratio > 0.0


def test_extract_clean_text_strips_ads_and_returns_title_body():
    html = """
    <html><head><title>My Niche Blog Post</title></head>
    <body>
        <article>
            <h1>Backyard Beekeeping for Beginners</h1>
            <p>Beekeeping is a rewarding hobby that supports local pollinators and
            can even yield your own honey harvest each season. Getting started
            requires a hive, protective gear, and a basic understanding of colony
            behavior throughout the year. New beekeepers should start with a single
            hive to learn the fundamentals before expanding their apiary.</p>
        </article>
        <div class="sponsored-banner">Buy discount beekeeping gear now!</div>
        <iframe src="https://doubleclick.net/ad"></iframe>
    </body></html>
    """
    title, body, ad_ratio = extract_clean_text(html, "https://example.com/post")
    assert title == "My Niche Blog Post"
    assert "beekeeping" in body.lower()
    assert "discount beekeeping gear" not in body
    assert ad_ratio > 0.0


def test_strip_ad_elements_survives_nested_tags_in_ad_containers():
    # Regression: decompose() on an ad container nulls its subtree's attrs, so
    # a child tag appearing later in the materialized find_all list crashed
    # with AttributeError ('NoneType' object has no attribute 'get').
    html = """
    <html><head><title>Nested Ad Page</title></head>
    <body>
        <div class="advertisement"><img src="ad.png"/></div>
        <article>
            <h1>Composting at Home</h1>
            <p>Composting turns kitchen scraps and yard waste into a rich soil
            amendment over a few months. A balanced pile needs a mix of green
            nitrogen-rich material and brown carbon-rich material, regular
            turning for aeration, and enough moisture to keep the microbes
            active without waterlogging the pile.</p>
        </article>
    </body></html>
    """
    soup, ad_ratio = strip_ad_elements(html)  # must not raise
    assert soup.find("div", class_="advertisement") is None
    assert soup.find("img") is None
    assert "Composting" in soup.get_text()

    title, body, _ = extract_clean_text(html, "https://example.com/compost")
    assert title == "Nested Ad Page"
    assert "composting" in body.lower()


from uaisearch.crawler import extract_links


def test_extract_links_resolves_relative_hrefs():
    html = '<a href="/about">About</a><a href="https://other.example/page">Other</a>'
    links = extract_links(html, "https://example.com/section/")
    assert "https://example.com/about" in links
    assert "https://other.example/page" in links


def test_extract_links_skips_non_http_and_dark_web_links():
    html = (
        '<a href="mailto:hi@example.com">Mail</a>'
        '<a href="javascript:void(0)">JS</a>'
        '<a href="http://exampleonionaddr.onion/page">Onion</a>'
        '<a href="/ok">Ok</a>'
    )
    links = extract_links(html, "https://example.com/")
    assert links == ["https://example.com/ok"]
