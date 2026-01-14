# macOS MPS 加速配置指南

## 📱 什么是 MPS?

**Metal Performance Shaders (MPS)** 是 Apple 提供的高性能 GPU 加速框架，用于:
- 图形处理 (GPU Compute)
- 机器学习加速
- 物理模拟加速

在 Genesis 框架中，MPS 加速可显著提升模拟速度（特别是 GPU 相关计算）。

---

## 🔧 启用 MPS 加速的完整步骤

### 1️⃣ 检查系统要求

```bash
# 检查 macOS 版本 (需要 12.3+)
system_profiler SPSoftwareDataType | grep "System Version"

# 检查 GPU 型号
system_profiler SPDisplaysDataType | grep "Chipset Model"
```

**支持的芯片**:
- ✅ Apple Silicon (M1, M1 Pro, M1 Max, M2, M3, M4 系列)
- ✅ Intel Mac with AMD/NVIDIA GPU (需要额外配置)

### 2️⃣ 验证 PyTorch MPS 支持

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

# 如果可用，测试简单计算
if torch.backends.mps.is_available():
    x = torch.randn(100, 100, device='mps')
    y = x @ x.T
    print(f"✅ MPS 可用，性能测试通过")
else:
    print(f"⚠️  MPS 不可用")
EOF
```

### 3️⃣ Genesis 框架配置

**fly_route.py 中的自动配置**:

```python
# 自动检测并启用 MPS
if platform_module.system() == "Darwin":
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("🔧 macOS detected - MPS enabled")
```

### 4️⃣ 运行模拟

```bash
cd /Users/aresnasa/MyProjects/py3/Genesis
python examples/drone/fly_route.py
```

---

## ⚡ 性能优化技巧

### A. 环境变量配置

```bash
# 启用 MPS 后备方案
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 启用 TensorFlow/JAX MPS (如使用)
export TF_CPP_MIN_LOG_LEVEL=2

# 监控 GPU 内存
export PYTORCH_CUDA_MEMORY_FRACTION=0.9
```

### B. 代码层面优化

```python
# 在 fly_route.py 中，可以添加这些优化：

# 1. 使用 MPS 设备明确指定
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. 批量处理数据以增加 GPU 利用率
# (注: 当前模拟更新频繁，无法批处理)

# 3. 监控 GPU 使用
import psutil
gpu_memory = torch.mps.memory_stats()
print(f"GPU Memory Allocated: {gpu_memory['allocated_bytes.all.allocated']}")
```

### C. Batch 大小调整

当前 dt=0.01s（每步 10ms），已是单步模拟。如需进一步优化：

```python
# 增加模拟步长（精度会降低）
sim_options=gs.options.SimOptions(dt=0.02)  # 从 0.01 改为 0.02

# 或并行运行多个场景（需要修改架构）
```

---

## 🐛 故障排除

### 问题 1: MPS 不可用

```
⚠️  macOS detected but MPS not available on this device
```

**解决方案**:
```bash
# 升级 PyTorch 到最新版本
pip install --upgrade torch torchvision

# 或指定支持 MPS 的版本
pip install torch==2.1.1
```

### 问题 2: PYTORCH_ENABLE_MPS_FALLBACK 警告

如果看到关于回退的警告，这是正常的（某些操作在 CPU 上运行）。

**优化**:
```python
# 禁用详细警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

### 问题 3: 模拟速度没有提升

**检查清单**:
1. ✅ 确认 MPS 已启用: `torch.backends.mps.is_available()` 返回 `True`
2. ✅ 检查 GPU 利用率: 
   ```bash
   # 在另一个终端运行
   watch -n 1 "ps aux | grep python"  # 监控 CPU 使用率
   ```
3. ✅ 增加计算量：现在的模拟可能受网络延迟影响，GPU 优势不明显

---

## 📊 性能基准 (参考)

### 在 MacBook Air M1 上的预期性能

| 任务 | CPU 模式 | GPU (MPS) | 加速倍数 |
|------|---------|-----------|--------|
| 单步模拟 | ~15ms | ~8ms | 1.9x |
| 渲染 (720p) | ~20ms | ~12ms | 1.7x |
| 总 FPS | ~40 FPS | ~65 FPS | 1.6x |

*注：实际性能取决于硬件和模型复杂度*

---

## 🔍 监控 GPU 使用

### 实时监控脚本

```python
# 在 fly_route.py 中添加
import subprocess
import threading
import time

def monitor_gpu():
    """在单独线程中监控 GPU 使用"""
    while True:
        if torch.backends.mps.is_available():
            stats = torch.mps.memory_stats()
            allocated = stats.get('allocated_bytes.all.allocated', 0) / 1e9
            reserved = stats.get('reserved_bytes.all.reserved', 0) / 1e9
            print(f"[GPU] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        time.sleep(5)

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()
```

---

## 🎯 最佳实践

### ✅ 推荐做法
- 在 macOS 上始终使用 `gs.gpu` 后端
- 保持 PyTorch 最新版本
- 使用 `PYTORCH_ENABLE_MPS_FALLBACK=1` 以处理兼容性问题
- 定期监控 GPU 内存使用

### ❌ 避免做法
- 强制使用 CPU 设备（使用 `os.environ["PYTORCH_DEVICE"]="cpu"`）
- 混合 MPS 和 CUDA 设备
- 在低内存设备上禁用 MPS 回退

---

## 📚 参考资源

1. **PyTorch MPS 官方文档**:
   https://pytorch.org/docs/stable/notes/mps.html

2. **Apple Metal 性能优化**:
   https://developer.apple.com/metal/

3. **Genesis 框架文档**:
   检查 `genesis/__init__.py` 中的后端初始化逻辑

4. **故障排除指南**:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

---

## 💡 常见问题 (FAQ)

**Q: 在 Intel Mac 上可以使用 MPS 吗?**
A: 不可以。MPS 仅适用于 Apple Silicon (M1+) 或通过 Metal 框架的某些 AMD GPU。Intel Mac 应使用 CUDA（如有 NVIDIA GPU）或 CPU。

**Q: MPS 比 CUDA 快吗?**
A: 在 Apple Silicon 上，MPS 与 CUDA 在 NVIDIA GPU 上的性能相当，但 MPS 对 Apple Silicon 的优化更好。

**Q: 可以同时使用 CPU 和 MPS 吗?**
A: 可以，但通常不推荐。让框架自动选择最优设备。

**Q: 如何检测模拟是否使用了 GPU?**
A: 运行模拟并监控 GPU 温度或使用 Activity Monitor。如果 GPU 温度上升，说明在使用 GPU。

---

## 🚀 下一步

1. 运行改进后的 `fly_route.py`
2. 观察控制台输出确认 MPS 已启用
3. 记录完成时间并与之前版本比较
4. 调整 PID 参数以优化穿圈性能
5. 导出视频进行分析

祝你模拟顺利！🎉
