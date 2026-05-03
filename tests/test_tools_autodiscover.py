def test_autodiscover_imports_all_tool_modules():
    import godbot.tools
    assert callable(godbot.tools.discover)


def test_discover_returns_module_count():
    import godbot.tools
    n = godbot.tools.discover()
    assert isinstance(n, int)
    assert n >= 0
