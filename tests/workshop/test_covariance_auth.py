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
