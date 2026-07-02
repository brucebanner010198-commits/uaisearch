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
