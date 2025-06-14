import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modern_edge_toolkit.detectors_modern import load_config, list_input_images


def test_load_config():
    cfg = load_config(Path('modern_edge_toolkit/config_modern.yaml'))
    assert 'input_dir' in cfg
    assert 'output_dir' in cfg


def test_list_input_images(tmp_path):
    img1 = tmp_path / 'a.jpg'
    img1.write_bytes(b'0')
    img2 = tmp_path / 'b.png'
    img2.write_bytes(b'0')
    (tmp_path / 'c.txt').write_text('x')
    images = list_input_images(tmp_path)
    assert img1 in images and img2 in images
    assert len(images) == 2
