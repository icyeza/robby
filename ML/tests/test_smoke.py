import os


def test_packages_import_and_telemetry_off() -> None:
    import robson_engine
    import robson_ml

    assert robson_engine.__version__ == "1.1.0"
    assert robson_ml.__version__ == "0.1.0"
    assert os.environ["DO_NOT_TRACK"] == "1"
