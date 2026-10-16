import pytest

# Six points on a line, the first three tagged 1 and the rest 2, so the k = 2
# nearest the origin are rowids 1 and 2 overall and rowids 4 and 5 among tag 2.
# A KNN query takes its filters before its k: whatever table a filter's value
# comes from, the rows are the k nearest among those that pass it.


@pytest.fixture()
def fill(db):
    db.execute(
        "create virtual table v using vec0("
        "part integer partition key, embedding float[2] distance_metric=l2, tag integer)"
    )
    db.executemany(
        "insert into v(rowid, part, embedding, tag) values (?, ?, ?, ?)",
        [[i, 1 if i <= 3 else 2, f"[{i},0]", 1 if i <= 3 else 2] for i in range(1, 7)],
    )
    db.execute("create table w(t integer, n integer, r real, q text)")
    db.execute("insert into w values (2, 2, 4.5, '[0,0]')")
    return db


def rowids(db, sql):
    return [row[0] for row in db.execute(sql)]


CONSTANT = "select v.rowid from v where v.embedding match '[0,0]' and v.k = 2"


@pytest.mark.parametrize("tables", ["w, v", "v, w"])
@pytest.mark.parametrize(
    "joined, constant",
    [
        ("v.tag = w.t", "v.tag = 2"),
        ("v.tag in (select w.t)", "v.tag in (2)"),
        ("v.part = w.t", "v.part = 2"),
        ("v.rowid in (select value from json_each('[4,5,6]') where value > w.t)", "v.rowid in (4, 5, 6)"),
        ("v.distance < w.r", "v.distance < 4.5"),
    ],
)
def test_a_filter_read_from_another_table_is_taken_before_the_k(fill, tables, joined, constant):
    db = fill
    knn = f"select v.rowid from {tables} where v.embedding match '[0,0]' and v.k = 2 and {joined}"
    assert rowids(db, knn) == rowids(db, f"{CONSTANT} and {constant}")


@pytest.mark.parametrize("tables", ["w, v", "v, w"])
def test_the_query_and_k_can_come_from_another_table(fill, tables):
    db = fill
    knn = f"select v.rowid from {tables} where v.embedding match w.q and v.k = w.n"
    assert rowids(db, knn) == [1, 2]
