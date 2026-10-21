# 模型权重目录

本目录用于存放 Spark-X2.5 的 GGUF 量化权重，程序启动时会自动从这里加载。

## 需要的文件

| 文件名 | 大小 | 用途 |
| --- | --- | --- |
| `Spark-X2.5-1.7B-Q4_K_M.gguf` | 1.03 GB | 极速模式（启动时自动加载） |
| `Spark-X2.5-4B-Q4_K_M.gguf` | 2.42 GB | 高精度模式（首次使用时加载） |

两个文件合计约 **3.45 GB**。

## 获取方式

在项目根目录运行下载脚本：

```bash
python scripts/download_models.py                # 两个模型
python scripts/download_models.py --model 1.7b   # 只要极速模型（约 1.03 GB）
python scripts/download_models.py --verify-only  # 校验已下载文件的完整性
```

脚本只依赖 Python 标准库（无需先 `pip install`），支持**断点续传**、
**SHA256 校验**，并会自动在 `huggingface.co` 不可达时切换到镜像源。

## 手动下载

从以下 Hugging Face 仓库获取（**Apache-2.0 许可**）：

- `XHToken/Spark-X2.5-1.7B-GGUF`
- `XHToken/Spark-X2.5-4B-GGUF`

若 `huggingface.co` 无法访问，可使用镜像 `https://hf-mirror.com`。

## 文件名可以不完全一致

程序查找权重时按以下顺序逐级放宽匹配：

1. 精确文件名；
2. 大小写不敏感（`spark-x2.5-1.7b-q4_k_m.gguf` 也能识别）；
3. 按参数量特征匹配（提取 `1.7b` / `4b`），因此从 Ollama 或
   LM Studio 下载的 `spark-x2.5-1.7b-q4_k_m-00001-of-00001.gguf`
   这类文件名也能被正确识别。

因此通常不需要手动改文件名。程序**不会**把 4B 的文件误当作 1.7B 加载。

## 其他可用位置

除本目录外，程序也会查找：

- `$PINGYITONG_MODELS_DIR`（环境变量指定的目录，优先级最高）
- `%LOCALAPPDATA%\PingYiTong\models\`（安装后的用户级数据目录）

## 关于版本控制

本目录的**权重文件不纳入版本控制**（见根目录 `.gitignore`）——
3.45 GB 不适合入库。只有本说明文件会被跟踪。

## 权限与合规

模型权重版权归其开源方所有，本项目不主张任何权利，也不修改权重本身。
使用时请遵守其 Apache-2.0 许可协议。详见项目根目录的 `LICENSE`。
