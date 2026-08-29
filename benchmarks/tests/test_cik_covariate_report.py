from benchmarks.cik.report_covariate_behavior import summarise


def test_report_separates_proposal_validation_and_admission():
    rows = [
        {"tables_proposed": 1, "rows_proposed": 2,
         "tables_validated": 1, "rows_validated": 2,
         "table_bound": True, "covariates_considered": True,
         "covariates_admitted": True, "route": "gnomon"},
        {"tables_proposed": 1, "rows_proposed": 1,
         "tables_validated": 0, "rows_validated": 0,
         "table_bound": False, "covariates_considered": False,
         "covariates_admitted": False, "route": "gnomon"},
        {"tables_proposed": 0, "rows_proposed": 0,
         "tables_validated": 0, "rows_validated": 0,
         "table_bound": False, "covariates_considered": False,
         "covariates_admitted": False, "route": "gnomon"},
    ]
    report = summarise(rows)
    assert report["proposal_cases"] == 2
    assert report["validated_cases"] == 1
    assert report["fold_admitted_cases"] == 1
    assert report["proposal_validation_rate"] == 0.5
    assert report["validation_to_admission_rate"] == 1.0
    assert report["all_routes_governed"] is True
