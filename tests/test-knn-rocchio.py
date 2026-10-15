import sqlite3

import pytest


def knn(db, sql, parameters):
    return [(row[0], round(row[1], 5)) for row in db.execute(sql, parameters)]


BASE = "select rowid, distance from v where embedding match ? and k = ?"


def fill(db, metric):
    db.execute(
        f"create virtual table v using vec0(embedding float[3] distance_metric={metric})"
    )
    db.executemany(
        "insert into v(rowid, embedding) values (?, ?)",
        [
            [1, "[1,0,0]"],
            [2, "[0,1,0]"],
            [3, "[0.7,0.7,0]"],
            [4, "[0,0,1]"],
            [5, "[-1,0,0]"],
        ],
    )


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_rocchio_moves_the_query_toward_the_feedback_row(db, metric):
    """rocchio = r searches q + 0.75 * r: the same rows and distances as that vector."""
    fill(db, metric)
    moved = knn(db, BASE + " and rocchio = ?", ["[1,0.2,0]", 5, 2])
    by_hand = knn(db, BASE, ["[1,0.95,0]", 5])
    assert moved == by_hand
    assert moved != knn(db, BASE, ["[1,0.2,0]", 5])


def test_rocchio_beta_weighs_the_feedback_row(db):
    fill(db, "cosine")
    assert knn(db, BASE + " and rocchio = ? and rocchio_beta = ?", ["[1,0.2,0]", 5, 2, 0.5]) == knn(
        db, BASE, ["[1,0.7,0]", 5]
    )
    assert knn(db, BASE + " and rocchio = ? and rocchio_beta = ?", ["[1,0.2,0]", 5, 2, 0.0]) == knn(
        db, BASE, ["[1,0.2,0]", 5]
    )


def test_rocchio_null_leaves_the_query_as_given(db):
    fill(db, "cosine")
    assert knn(db, BASE + " and rocchio = ?", ["[1,0.2,0]", 5, None]) == knn(
        db, BASE, ["[1,0.2,0]", 5]
    )


def test_rocchio_per_outer_row(db):
    """Each outer row's query moves by its own feedback row, in one statement."""
    fill(db, "cosine")
    db.execute("create table q(id integer primary key, vec text, fb integer)")
    db.executemany(
        "insert into q values (?, ?, ?)",
        [[10, "[1,0.2,0]", 2], [20, "[1,0.2,0]", 4], [30, "[1,0.2,0]", None]],
    )
    rows = db.execute(
        "select q.id, v.rowid from q join v on v.embedding match q.vec and v.k = 1"
        " and v.rocchio = q.fb order by q.id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [(10, 3), (20, 1), (30, 1)]


def test_rocchio_with_text_primary_key(db):
    db.execute(
        "create virtual table v using vec0(id text primary key, embedding float[3] distance_metric=cosine)"
    )
    db.executemany(
        "insert into v(id, embedding) values (?, ?)",
        [["a", "[1,0,0]"], ["b", "[0,1,0]"], ["c", "[0.7,0.7,0]"]],
    )
    moved = [
        (r[0], round(r[1], 5))
        for r in db.execute(
            "select id, distance from v where embedding match ? and k = 3 and rocchio = ?",
            ["[1,0.2,0]", "b"],
        )
    ]
    by_hand = [
        (r[0], round(r[1], 5))
        for r in db.execute(
            "select id, distance from v where embedding match ? and k = 3", ["[1,0.95,0]"]
        )
    ]
    assert moved == by_hand
    with pytest.raises(sqlite3.OperationalError, match="No row has the id given for rocchio"):
        db.execute(
            "select id from v where embedding match ? and k = 3 and rocchio = ?",
            ["[1,0.2,0]", "zz"],
        ).fetchall()


def test_rocchio_composes_with_mmr(db):
    fill(db, "cosine")
    assert knn(
        db, BASE + " and rocchio = ? and mmr_lambda = ?", ["[1,0.2,0]", 3, 2, 0.5]
    ) == knn(db, BASE + " and mmr_lambda = ?", ["[1,0.95,0]", 3, 0.5])


def test_rocchio_errors(db):
    fill(db, "cosine")
    with pytest.raises(sqlite3.OperationalError, match="Could not find a row with rowid 99"):
        db.execute(BASE + " and rocchio = ?", ["[1,0.2,0]", 3, 99]).fetchall()
    with pytest.raises(sqlite3.OperationalError, match="rocchio value in knn query must be a rowid"):
        db.execute(BASE + " and rocchio = ?", ["[1,0.2,0]", 3, "two"]).fetchall()
    with pytest.raises(sqlite3.OperationalError, match="rocchio_beta value in knn query must be finite"):
        db.execute(
            BASE + " and rocchio = ? and rocchio_beta = ?", ["[1,0.2,0]", 3, 2, float("inf")]
        ).fetchall()
    with pytest.raises(sqlite3.OperationalError, match="hidden \"rocchio\" column"):
        db.execute("insert into v(rowid, embedding, rocchio) values (9, '[1,1,1]', 1)")
    with pytest.raises(sqlite3.OperationalError, match="hidden \"rocchio_beta\" column"):
        db.execute("insert into v(rowid, embedding, rocchio_beta) values (9, '[1,1,1]', 1)")


def test_rocchio_needs_float_vectors(db):
    db.execute("create virtual table v using vec0(embedding int8[3] distance_metric=cosine)")
    db.execute("insert into v(rowid, embedding) values (1, vec_int8('[100,0,0]'))")
    db.execute("insert into v(rowid, embedding) values (2, vec_int8('[0,100,0]'))")
    with pytest.raises(sqlite3.OperationalError, match="rocchio feedback needs a float32 vector column"):
        db.execute(
            "select rowid from v where embedding match vec_int8('[100,20,0]') and k = 2 and rocchio = 2"
        ).fetchall()
