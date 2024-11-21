import scanreader
from pathlib import Path

data = Path().home() / 'caiman_data' / 'raw' / 'demo_data.tiff'
scan = scanreader.read_scan()
