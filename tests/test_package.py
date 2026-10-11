def test_package_imports():
    import everyngram
    from everyngram import decode, memory

    assert everyngram.__name__ == "everyngram"
    assert memory is not None
    assert decode is not None


def test_softmax_is_defined_only_in_fuse():
    """softmax 只应存在于 everyngram.fuse,decode 不再转发别名(审计 P6)。"""
    import everyngram.decode as decode
    import everyngram.fuse as fuse

    assert not hasattr(decode, "_softmax")
    assert callable(fuse.softmax)
