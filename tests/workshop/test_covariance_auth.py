"""Authenticated Drive downloads use prepared credentials, never worker OAuth."""

from types import SimpleNamespace
from unittest.mock import Mock


def test_drive_download_uses_credentials_and_resource_header(tmp_path, monkeypatch, capsys):
    import google.auth
    import googleapiclient.discovery
    import googleapiclient.http

    from src.workshop.covariance import download_colab

    credentials = SimpleNamespace(token="do-not-print")
    auth = Mock(return_value=(credentials, "project"))
    monkeypatch.setattr(google.auth, "default", auth)
    request = SimpleNamespace(headers={})
    drive = Mock()
    drive.files.return_value.get_media.return_value = request
    build = Mock(return_value=drive)
    monkeypatch.setattr(googleapiclient.discovery, "build", build)

    class Download:
        def __init__(self, output, actual_request):
            assert actual_request is request
            self.output = output

        def next_chunk(self):
            self.output.write(b"matrix")
            return SimpleNamespace(progress=lambda: 1.), True

    monkeypatch.setattr(googleapiclient.http, "MediaIoBaseDownload", Download)
    path = tmp_path / "cov.pt"
    download_colab("file_id", path, "resource_key")
    assert path.read_bytes() == b"matrix"
    assert request.headers == {"X-Goog-Drive-Resource-Keys": "file_id/resource_key"}
    build.assert_called_once_with("drive", "v3", credentials=credentials, cache_discovery=False)
    assert "do-not-print" not in capsys.readouterr().out


def test_unchanged_link_reuses_covariance_and_new_link_replaces_it(tmp_path, monkeypatch):
    from src.workshop import covariance
    from src.workshop.setup import Environment

    monkeypatch.delenv("LATIUM_COV_PATH", raising=False)
    download = Mock(side_effect=lambda file_id, path: path.write_bytes(file_id.encode()))
    monkeypatch.setattr(covariance, "download_public", download)
    environment = Environment(tmp_path)
    original = "original_drive_id"
    replacement = "replacement_drive_id"
    path = covariance.ensure_covariance(environment, original, mode="public", manual_upload=False)
    assert covariance.ensure_covariance(environment, original, mode="public", manual_upload=False) == path
    assert path.read_bytes() == original.encode()
    assert download.call_count == 1
    assert covariance.ensure_covariance(environment, replacement, mode="public", manual_upload=False) == path
    assert path.read_bytes() == replacement.encode()
    assert download.call_count == 2
