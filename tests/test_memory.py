import pytest

from everyngram.memory import Memory


def test_build_rejects_negative_budget():
    with pytest.raises(ValueError, match="budget"):
        Memory.build(
            [1, 2, 3, 4],
            fingerprint="fp",
            orders=(1,),
            budget=-5,
        )

# 长度为 9。注意最后一个 token 不能作为键,因为它后面没有东西。
CORPUS = [1, 2, 3, 1, 2, 3, 1, 2, 4]


def test_build_counts_and_next_distribution():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1,), budget=100, continuation_len=4)

    assert set(mem.entries) == {(1,), (2,), (3,)}
    assert mem.entries[(1,)].count == 3
    assert mem.entries[(2,)].count == 3
    assert mem.entries[(3,)].count == 2
    assert mem.entries[(1,)].next_top == ((2, 3),)
    assert mem.entries[(2,)].next_top == ((3, 2), (4, 1))
    assert mem.entries[(3,)].next_top == ((1, 2),)


def test_build_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        Memory.build([1, 2], fingerprint="fp", orders=(1, 2, 3))


def test_lookup_prefers_longest_key():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)

    key, entry = mem.lookup([1, 2])
    assert key == (1, 2)
    assert entry.count == 3

    key, entry = mem.lookup([3, 1, 2])
    assert key == (3, 1, 2)
    assert entry.count == 2


def test_lookup_falls_back_to_shorter_key():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)

    # (9, 1, 2) 不是表里的键,但 (1, 2) 是
    key, entry = mem.lookup([9, 1, 2])
    assert key == (1, 2)


def test_lookup_returns_none_on_miss():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    assert mem.lookup([4]) is None


def test_draft_uses_first_occurrence_continuation():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(2,), budget=100, continuation_len=6)

    # (1, 2) 首次出现在下标 0,后续从下标 2 开始:3, 1, 2, 3, 1, 2
    assert mem.draft([1, 2], gamma=3) == (3, 1, 2)


def test_draft_truncates_to_gamma():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(2,), budget=100, continuation_len=6)
    assert mem.draft([1, 2], gamma=1) == (3,)


def test_draft_returns_empty_on_miss_and_zero_gamma():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(2,), budget=100)
    assert mem.draft([4], gamma=3) == ()
    assert mem.draft([1, 2], gamma=0) == ()


def test_budget_truncates_and_reports_dropped():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=2)

    assert len(mem.entries) == 2
    # 三个阶各 3 个键,共 9 个,保留 2 个
    assert mem.stats()["dropped"] == 7
    # 保留的是最高阶,因为排序键是 (阶数, 频次)
    assert all(len(key) == 3 for key in mem.entries)


def test_stats_counts_by_order():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    stats = mem.stats()

    assert stats["entries"] == 9
    assert stats["dropped"] == 0
    assert stats["by_order"] == {"1": 3, "2": 3, "3": 3}


def test_save_load_roundtrip(tmp_path):
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    path = tmp_path / "m.ngmem"
    mem.save(path)
    back = Memory.load(path)

    assert back.entries == mem.entries
    assert back.orders == mem.orders
    assert back.fingerprint == "fp"
    assert back.continuation_len == mem.continuation_len
    assert back.stats() == mem.stats()


def test_save_is_byte_identical(tmp_path):
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    a = tmp_path / "a.ngmem"
    b = tmp_path / "b.ngmem"
    mem.save(a)
    Memory.load(a).save(b)

    assert a.read_bytes() == b.read_bytes()


def test_load_rejects_wrong_version(tmp_path):
    path = tmp_path / "bad.ngmem"
    path.write_text('{"format_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError):
        Memory.load(path)


def test_subset_keeps_only_given_keys_and_accounts_dropped():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    keys = list(mem.entries)[:4]

    sub = mem.subset(keys, budget=4)

    assert set(sub.entries) == set(keys)
    assert sub.budget == 4
    assert sub.stats()["dropped"] == mem.stats()["entries"] - 4
    assert sub.fingerprint == mem.fingerprint
    assert sub.orders == mem.orders
    assert sub.continuation_len == mem.continuation_len
    assert sub.next_top_k == mem.next_top_k


def test_subset_rejects_unknown_keys():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1,), budget=100)

    with pytest.raises(KeyError):
        mem.subset([(999,)])


def test_subset_without_budget_uses_kept_count():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    keys = list(mem.entries)[:2]

    sub = mem.subset(keys)

    assert sub.budget == 2
    assert len(sub.entries) == 2
