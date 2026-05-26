# 1-tBu + H2CO

这个目录用于 RTIP/JAX 的 `synthesize --inputs` 入口。

- `1.xyz`: 第一个反应物
- `2.xyz`: 第二个反应物
- `product.xyz`: SI 中的优化产物结构，如果存在
- `ts.xyz`: SI 中的过渡态结构，如果存在

示例：

```bash
cd JAX
python -m rtip_jax.cli synthesize --inputs examples/ic5c02384/reactions/1-tBu__H2CO/1.xyz examples/ic5c02384/reactions/1-tBu__H2CO/2.xyz --output examples/ic5c02384/reactions/1-tBu__H2CO/IS.xyz --dist 5.0 --seed 0
```

后续如果有可用 DeePMD 模型，可以把生成的 `IS.xyz` 作为 `deepmd-pathway` 的 `--input`。
