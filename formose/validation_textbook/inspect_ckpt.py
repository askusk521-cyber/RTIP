"""读取 DPA-3.2-5M checkpoint 元数据（zip 内 data.pkl），打印配置。"""

from __future__ import annotations

import io
import sys
import zipfile

import torch


def main() -> None:
    model = sys.argv[1]
    with zipfile.ZipFile(model) as zf:
        pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        raw = zf.read(pkl_name)
    buf = io.BytesIO(raw)
    obj = torch.load(buf, map_location="cpu", weights_only=False)
    print("type:", type(obj))
    if isinstance(obj, dict):
        for key in obj:
            val = obj[key]
            if isinstance(val, str) and len(val) < 5000:
                print(f"--- {key} ---")
                print(val)
            elif isinstance(val, (dict, list)) and len(str(val)) < 5000:
                print(f"--- {key} ---")
                print(val)
            else:
                print(f"--- {key}: {type(val)} len={len(str(val))} ---")


if __name__ == "__main__":
    main()
