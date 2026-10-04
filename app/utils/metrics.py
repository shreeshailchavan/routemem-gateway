try:
    from prometheus_client import Counter, Histogram, Gauge

    REQUEST_COUNT = Counter(
        "routemem_requests_total",
        "Total number of HTTP requests processed by RouteMem",
        ["method", "endpoint", "status"]
    )

    CACHE_HITS = Counter(
        "routemem_cache_hits_total",
        "Total number of cache hits",
        ["cache_type"]
    )

    TTFT_HISTOGRAM = Histogram(
        "routemem_ttft_seconds",
        "Time to first token in seconds",
        buckets=[0.002, 0.005, 0.015, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )

    TOKEN_REDUCTION_GAUGE = Gauge(
        "routemem_token_reduction_ratio",
        "Latest prompt token compression ratio"
    )

    COST_SAVINGS_TOTAL = Counter(
        "routemem_cost_savings_usd_total",
        "Cumulative estimated USD saved by RouteMem cache and dynamic routing"
    )

except ImportError:
    class MockMetric:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass

    REQUEST_COUNT = MockMetric()
    CACHE_HITS = MockMetric()
    TTFT_HISTOGRAM = MockMetric()
    TOKEN_REDUCTION_GAUGE = MockMetric()
    COST_SAVINGS_TOTAL = MockMetric()
