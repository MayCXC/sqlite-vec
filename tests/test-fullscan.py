import struct


def _f32(values):
    return struct.pack(f"{len(values)}f", *values)


def test_fullscan_reads_every_vector_across_chunks(db):
    # Two vector columns, so each keeps its own handle, over chunks with gaps.
    db.execute(
        "create virtual table v using vec0(a float[3], b float[2], chunk_size=8)"
    )
    for i in range(1, 51):
        db.execute(
            "insert into v(rowid, a, b) values (?, ?, ?)",
            [i, _f32([i, i + 0.5, -i]), _f32([2 * i, 1 / i])],
        )
    db.execute("delete from v where rowid % 7 = 0")

    scanned = {r[0]: (r[1], r[2]) for r in db.execute("select rowid, a, b from v")}
    assert len(scanned) == 50 - 7
    for rowid, vectors in scanned.items():
        point = db.execute("select a, b from v where rowid = ?", [rowid]).fetchone()
        assert vectors == (point[0], point[1])
        assert vectors[0] == _f32([rowid, rowid + 0.5, -rowid])


def test_fullscan_reads_a_chunk_rewritten_during_the_scan(db):
    # Rewriting a chunk's row expires a blob handle open on it; the scan reads
    # the rest of the chunk from the new row, not from what it held before.
    db.execute("create virtual table v using vec0(a float[2], chunk_size=8)")
    for i in range(1, 17):
        db.execute("insert into v(rowid, a) values (?, ?)", [i, _f32([i, 0])])
    first = db.execute("select min(chunk_id) from v_chunks").fetchone()[0]

    scan = db.execute("select rowid, a from v")
    head = [scan.fetchone() for _ in range(2)]
    rewritten = b"".join(_f32([100 + i, 1]) for i in range(1, 9))
    db.execute(
        "update v_vector_chunks00 set vectors = ? where rowid = ?", [rewritten, first]
    )
    rest = scan.fetchall()

    rows = {r[0]: struct.unpack("2f", r[1]) for r in head + rest}
    assert sorted(rows) == list(range(1, 17))
    # Python's cursor may have stepped to the third row before the update, so
    # the rows checked are those it can only have read after it.
    for rowid in range(4, 9):
        assert rows[rowid] == (100 + rowid, 1)
    for rowid in range(9, 17):
        assert rows[rowid] == (rowid, 0)
