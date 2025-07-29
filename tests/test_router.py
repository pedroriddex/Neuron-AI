import time
from talamo.router import TalamoRouter, Expert


def test_round_robin_and_latency():
    router = TalamoRouter()
    seen = []

    for _ in range(6):
        start = time.perf_counter_ns()
        experts = router.route_input("hello")
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        # exactly one expert returned
        assert len(experts) == 1
        assert isinstance(experts[0], Expert)
        assert elapsed_ms < 5  # acceptance criterion
        seen.append(experts[0])

    # Round-robin cycles through experts in order
    cycle = list(Expert) * 2  # two full cycles of 3 experts
    assert seen == cycle[:6]
