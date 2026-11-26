from tests.conftest import make_pdf


def test_identical_file_is_not_indexed_twice(docs, store):
    pdf = make_pdf()
    first = docs.process_upload("P0001", "a.pdf", pdf)
    count = store.collection.count()
    again = docs.process_upload("P0001", "renamed.pdf", pdf)
    assert again.status == "duplicate"
    assert store.collection.count() == count == first.chunks


def test_changed_report_requires_confirmation(docs, store, tmp_path):
    docs.process_upload("P0001", "a.pdf", make_pdf(hemoglobin="12.2"))
    count = store.collection.count()
    res = docs.process_upload("P0001", "a2.pdf", make_pdf(hemoglobin="14.0"))
    assert res.status == "needs_confirmation"
    assert store.collection.count() == count                          # untouched
    assert len(list((tmp_path / "uploads").rglob("*.pdf"))) == 1       # staged file removed


def test_confirmed_replacement_keeps_history(docs, store):
    old = docs.process_upload("P0001", "a.pdf", make_pdf(hemoglobin="12.2"))
    new = docs.process_upload("P0001", "a2.pdf", make_pdf(hemoglobin="14.0"), allow_replace=True)
    assert new.status == "replaced" and new.version == 2
    assert new.superseded_upload_ids == [old.upload_id]

    reports = docs.list_reports("P0001")
    assert len(reports) == 2                                           # old version preserved
    assert {(r.version, r.is_active) for r in reports} == {(1, False), (2, True)}
    assert len(docs.list_reports("P0001", include_superseded=False)) == 1
    assert old.stored_path != new.stored_path


def test_same_report_id_for_different_patients_is_independent(docs):
    assert docs.process_upload("P0001", "a.pdf", make_pdf("P0001", "R0001")).status == "indexed"
    assert docs.process_upload("P0002", "b.pdf", make_pdf("P0002", "R0001")).status == "indexed"
