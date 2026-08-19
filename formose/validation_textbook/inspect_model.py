"""检查 DPA-3.2-5M 模型的多域头配置与推理域选择。"""

from __future__ import annotations

import json
import sys


def main() -> None:
    from deepmd.infer import DeepPot

    model = sys.argv[1]
    dp = DeepPot(model)
    print("DeepPot object:", type(dp))
    attrs = [a for a in dir(dp) if not a.startswith("_")]
    print("attrs:", attrs)

    m = getattr(dp, "model", None)
    print("model attr type:", type(m))
    if m is not None:
        for attr in ("config", "model_config", "head", "heads", "domain", "type_map"):
            val = getattr(m, attr, "<missing>")
            if val != "<missing>":
                try:
                    print(f"{attr}:", json.dumps(val, indent=1)[:4000])
                except TypeError:
                    print(f"{attr}:", val)

    print("=== get_model_def_script ===")
    try:
        script = dp.get_model_def_script()
        print(script[:6000])
    except Exception as exc:  # noqa: BLE001
        print("get_model_def_script failed:", exc)

    # 尝试查看底层 pt 文件的 config 文本（.pt 为 zip 容器）
    import zipfile

    if model.endswith(".pt") and zipfile.is_zipfile(model):
        with zipfile.ZipFile(model) as zf:
            names = zf.namelist()
            print("zip entries:", names[:20])
            for candidate in ("config.json", "model/config.json", "config"):
                if candidate in names:
                    print(f"--- {candidate} ---")
                    print(zf.read(candidate).decode("utf-8", "replace")[:4000])


if __name__ == "__main__":
    main()
