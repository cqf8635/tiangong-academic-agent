from types import SimpleNamespace

import radar_sync_engine as mod


def test_background_sync_stops_after_circuit_breaker_activates(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "RADAR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    class Searcher:
        cooldown = 0.0
        calls = 0

        def is_user_active(self, _window=30.0):
            return False

        def arxiv_cooldown_remaining(self):
            return self.cooldown

        def search_radar(self, **_kwargs):
            self.calls += 1
            self.cooldown = 60.0
            return {"papers": [{"paper_id": "test"}]}

    searcher = Searcher()
    engine = mod.RadarSyncEngine(searcher)
    engine.sync_core_majors_now()

    assert searcher.calls == 1


def test_scheduler_does_not_start_full_sync_immediately_on_process_start(monkeypatch):
    engine = mod.RadarSyncEngine(SimpleNamespace())
    calls = []
    monkeypatch.setattr(engine, "sync_core_majors_now", lambda: calls.append("sync"))

    def stop_on_first_wait(_seconds):
        engine._stop_event.set()

    monkeypatch.setattr(mod.time, "sleep", stop_on_first_wait)
    engine._background_loop()

    assert calls == []
