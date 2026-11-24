# pairwise-lagcomp-rewind-hit

## 领域

游戏、图形与多媒体

## 主题

延迟补偿命中回溯

## 说明

时间戳索引、插值钳制与请求隔离。

初始状态下 `gameplay/lagcomp_rewind_hit.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
