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
