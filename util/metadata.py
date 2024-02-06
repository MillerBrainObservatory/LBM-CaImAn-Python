import re
import json


def _parse_value(value_str):
    if value_str.startswith("'") and value_str.endswith("'"):
        return value_str[1:-1]
    if value_str == 'true':
        return True
    if value_str == 'false':
        return False
    if value_str == 'NaN':
        return float('nan')
    if value_str == 'Inf':
        return float('inf')
    if re.match(r'^\d+(\.\d+)?$', value_str):
        return float(value_str) if '.' in value_str else int(value_str)
    if re.match(r'^\[(.*)]$', value_str):
        return [_parse_value(v.strip()) for v in value_str[1:-1].split()]
    return value_str


def _parse_key_value(parse_line):
    key_str, value_str = parse_line.split(' = ', 1)
    return key_str, _parse_value(value_str)


def parse(metadata_str):
    """
    Parses the metadata string from a ScanImage Tiff file.

    :param metadata_str:
    :return metadata_kv, metadata_json:
    """
    lines = metadata_str.split('\n')
    metadata_kv = {}
    json_portion = []
    parsing_json = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('SI.'):
            key, value = _parse_key_value(line)
            metadata_kv[key] = value
        elif line.startswith('{'):
            parsing_json = True
        if parsing_json:
            json_portion.append(line)
    metadata_json = json.loads('\n'.join(json_portion))
    return metadata_kv, metadata_json
