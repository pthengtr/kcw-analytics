from pathlib import Path
import tarfile

from src.kcw.archive_outputs import write_tar_gz


def test_write_tar_gz_skips_desktop_ini(tmp_path: Path):
    src = tmp_path / "04_outputs"
    src.mkdir()
    (src / "report.csv").write_text("a,b\n")
    (src / "desktop.ini").write_text("junk")
    dest = tmp_path / "export.tar.gz"
    write_tar_gz(src, dest)
    with tarfile.open(dest, "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith("report.csv") for n in names)
    assert not any(n.endswith("desktop.ini") for n in names)
