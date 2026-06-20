# 质量门禁

合并前至少应通过以下检查：

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```

涉及真实机器人、部署配置或硬件链路的改动，还应按照 [部署矩阵](../operations/deployment-matrix.md) 和对应部署文档完成 inspection 和 diagnosis。
