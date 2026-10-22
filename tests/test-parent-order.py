import random
import sqlite3

import pytest

# A vector column with parent_order=<metadata column> keeps every row's parent
# in the _parentNN shadow table: the nearest row of its partition, by the
# column's distance, whose value in the order column is smaller. Of equally near
# rows the parent is the one latest in order, then by rowid. These tests compare
# the table with that definition computed by brute force, through vec0's own
# distance functions, after every kind of change.

METRICS = {"l2": "vec_distance_l2", "cosine": "vec_distance_cosine"}


def create(db, metric="l2", order_type="integer", partition=True, chunk_size=8):
    db.execute(
        "create virtual table v using vec0("
        + ("p integer partition key, " if partition else "")
        + f"e float[3] distance_metric={metric} parent_order=t, "
        f"t {order_type}, chunk_size={chunk_size})"
    )


def entries(db, table="v_parent00"):
    return {
        row[0]: (row[1], row[2])
        for row in db.execute(f"select rowid, parent, distance from {table}")
    }


def expected(db, metric="l2", partition=True):
    rows = db.execute(
        "select rowid, e, t" + (", p" if partition else ", null") + " from v"
    ).fetchall()
    distance = METRICS[metric]
    parents = {}
    for rowid, e, t, p in rows:
        best = None
        for q_rowid, q_e, q_t, q_p in rows:
            if q_p != p or not q_t < t:
                continue
            d = db.execute(f"select {distance}(?, ?)", [e, q_e]).fetchone()[0]
            key = (d, -q_t, -q_rowid)
            if best is None or key < best[0]:
                best = (key, q_rowid, d)
        if best:
            parents[rowid] = (best[1], best[2])
    return parents


def column(db, sql):
    return [row[0] for row in db.execute(sql)]


def check(db, metric="l2", partition=True):
    assert entries(db) == expected(db, metric, partition)
    assert column(db, "pragma integrity_check") == ["ok"]


def vector(rng):
    # Few distinct coordinates, so that equal distances and copies are common.
    return "[" + ",".join(str(rng.randint(0, 2)) for _ in range(3)) + "]"


def order(rng, order_type):
    t = rng.randint(0, 12)
    return t + 0.5 if order_type == "float" else t


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("order_type", ["integer", "float"])
@pytest.mark.parametrize("partition", [True, False])
def test_every_change_keeps_the_definition(db, metric, order_type, partition):
    rng = random.Random(f"{metric} {order_type} {partition}")
    create(db, metric, order_type, partition)

    def insert(rowid, replace=False):
        db.execute(
            ("insert or replace" if replace else "insert")
            + " into v(rowid, "
            + ("p, " if partition else "")
            + "e, t) values (?, "
            + ("?, " if partition else "")
            + "?, ?)",
            [rowid]
            + ([rng.randint(1, 2)] if partition else [])
            + [vector(rng), order(rng, order_type)],
        )

    rowids = list(range(1, 41))
    rng.shuffle(rowids)
    for rowid in rowids:
        insert(rowid)
    check(db, metric, partition)

    for rowid in rng.sample(rowids, 12):
        db.execute("delete from v where rowid = ?", [rowid])
        rowids.remove(rowid)
        check(db, metric, partition)

    for rowid in rng.sample(rowids, 8):
        db.execute("update v set e = ? where rowid = ?", [vector(rng), rowid])
        check(db, metric, partition)
    for rowid in rng.sample(rowids, 8):
        db.execute(
            "update v set t = ? where rowid = ?", [order(rng, order_type), rowid]
        )
        check(db, metric, partition)

    for rowid in rng.sample(rowids, 6):
        insert(rowid, replace=True)
        check(db, metric, partition)


def test_copies_link_each_to_the_copy_before_it(db):
    create(db, partition=False)
    for rowid, t in [(1, 10), (2, 20), (3, 30), (4, 40)]:
        db.execute("insert into v(rowid, e, t) values (?, '[1,1,1]', ?)", [rowid, t])
    assert entries(db) == {2: (1, 0.0), 3: (2, 0.0), 4: (3, 0.0)}
    # At equal order values, the larger rowid is the later row.
    db.execute("insert into v(rowid, e, t) values (5, '[1,1,1]', 30)")
    assert entries(db)[4] == (5, 0.0)
    check(db, partition=False)


def test_rows_of_equal_order_are_not_each_others_parents(db):
    create(db, partition=False)
    db.executemany(
        "insert into v(rowid, e, t) values (?, ?, 7)",
        [[1, "[1,0,0]"], [2, "[1,0,0]"], [3, "[0,1,0]"]],
    )
    assert entries(db) == {}
    check(db, partition=False)


def test_an_earlier_insert_becomes_the_parent_of_later_rows(db):
    create(db, partition=False)
    db.execute("insert into v(rowid, e, t) values (1, '[0,0,9]', 10)")
    db.execute("insert into v(rowid, e, t) values (2, '[1,0,0]', 20)")
    db.execute("insert into v(rowid, e, t) values (3, '[1,0,0]', 30)")
    assert entries(db) == {2: (1, pytest.approx(9.055385)), 3: (2, 0.0)}
    db.execute("insert into v(rowid, e, t) values (4, '[1,0,1]', 5)")
    assert entries(db) == {
        1: (4, pytest.approx(8.062258)),
        2: (4, 1.0),
        3: (2, 0.0),
    }
    check(db, partition=False)


def test_partitions_keep_separate_trees(db):
    create(db)
    db.executemany(
        "insert into v(rowid, p, e, t) values (?, ?, ?, ?)",
        [[1, 1, "[0,0,0]", 1], [2, 2, "[0,0,0]", 2], [3, 1, "[1,0,0]", 3]],
    )
    assert entries(db) == {3: (1, 1.0)}
    check(db)


def test_a_null_partition_key_is_one_partition(db):
    create(db)
    db.executemany(
        "insert into v(rowid, p, e, t) values (?, ?, ?, ?)",
        [[1, None, "[0,0,0]", 1], [2, None, "[1,0,0]", 2], [3, 5, "[0,0,0]", 3]],
    )
    assert entries(db) == {2: (1, 1.0)}
    assert column(db, "pragma integrity_check") == ["ok"]


def test_only_the_vector_column_with_parent_order_keeps_parents(db):
    db.execute(
        "create virtual table v using vec0("
        "a float[1], b float[1] parent_order=t, c float[1] parent_order=u, "
        "t integer, u integer)"
    )
    db.executemany(
        "insert into v(rowid, a, b, c, t, u) values (?, ?, ?, ?, ?, ?)",
        [[1, "[0]", "[0]", "[0]", 1, 2], [2, "[5]", "[1]", "[3]", 2, 1]],
    )
    assert column(
        db,
        "select name from sqlite_schema where type = 'table' "
        "and name like 'v_parent%' order by 1",
    ) == ["v_parent01", "v_parent02"]
    assert entries(db, "v_parent01") == {2: (1, 1.0)}
    assert entries(db, "v_parent02") == {1: (2, 3.0)}
    assert column(db, "pragma integrity_check") == ["ok"]


@pytest.mark.parametrize(
    "columns, message",
    [
        (
            "e float[1] parent_order=t",
            'parent_order of the "e" column names "t", which is not a metadata column',
        ),
        (
            "e float[1] parent_order=t, t text",
            'parent_order of the "e" column needs an integer or float metadata column',
        ),
        (
            "e float[1] parent_order=t, t boolean",
            'parent_order of the "e" column needs an integer or float metadata column',
        ),
        (
            "e float[1] parent_order=t, t integer partition key",
            'parent_order of the "e" column names "t", which is not a metadata column',
        ),
        ("e float[1] parent_order=t parent_order=t, t integer", "could not parse"),
        ("e float[1] parent_order, t integer", "could not parse"),
        ("e float[1] parent_order=1, t integer", "could not parse"),
    ],
)
def test_parent_order_names_an_integer_or_float_metadata_column(db, columns, message):
    with pytest.raises(sqlite3.OperationalError, match=message):
        db.execute(f"create virtual table v using vec0({columns})")


def test_the_parent_table_is_a_shadow_table(db):
    create(db, partition=False)
    db.execute("insert into v(rowid, e, t) values (1, '[0,0,0]', 1)")
    db.execute("insert into v(rowid, e, t) values (2, '[1,0,0]', 2)")
    db.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
    with pytest.raises(sqlite3.OperationalError, match="may not be modified"):
        db.execute("delete from v_parent00")
    assert entries(db) == {2: (1, 1.0)}


def test_rename_and_drop_carry_the_parent_table(db):
    create(db, partition=False)
    db.execute("insert into v(rowid, e, t) values (1, '[0,0,0]', 1)")
    db.execute("alter table v rename to w")
    db.execute("insert into w(rowid, e, t) values (2, '[1,0,0]', 2)")
    assert entries(db, "w_parent00") == {2: (1, 1.0)}
    assert column(
        db, "select name from sqlite_schema where tbl_name = 'w_parent00' order by 1"
    ) == ["sqlite_autoindex_w_parent00_1", "w_parent00"]
    assert column(db, "pragma integrity_check") == ["ok"]
    db.execute("drop table w")
    assert column(db, "select name from sqlite_schema where name like '%w%'") == []


def test_optimize_keeps_the_tree(db):
    create(db, partition=False)
    rng = random.Random(7)
    db.executemany(
        "insert into v(rowid, e, t) values (?, ?, ?)",
        [[i, vector(rng), order(rng, "integer")] for i in range(1, 30)],
    )
    db.execute("delete from v where rowid % 3 = 0")
    before = entries(db)
    db.execute("insert into v(v) values ('optimize')")
    assert entries(db) == before
    check(db, partition=False)


def test_integrity_check_reports_entries_that_break_the_definition(db):
    create(db, partition=False)
    db.executemany(
        "insert into v(rowid, e, t) values (?, ?, ?)",
        [[1, "[0,0,0]", 1], [2, "[1,0,0]", 2], [3, "[2,0,0]", 3], [4, "[3,0,0]", 4]],
    )
    db.execute("update v_parent00 set distance = 0.5 where rowid = 2")
    db.execute("update v_parent00 set parent = 4 where rowid = 3")
    db.execute("delete from v_parent00 where rowid = 4")
    db.execute("insert into v_parent00 values (9, 1, 1.0)")
    db.execute("insert into v_parent00 values (1, 8, 1.0)")
    report = db.execute("pragma integrity_check").fetchone()[0]
    assert report.splitlines() == [
        "In vec0 main.v:",
        "Parent 8 of row 1 is not a row",
        "Row 2 is at distance 1 from its parent 1, not the 0.5 in %_parent00",
        "Parent 4 of row 3 is not earlier than it",
        "Row 4 has no entry in %_parent00, but an earlier row in its partition",
        "Wrong number of entries in %_parent00 table - expected 3, actual 4",
    ]
