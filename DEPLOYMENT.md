# Navilab 云服务器部署与版本兼容说明

## 一、当前环境适用的 Isaac Sim / Isaac Lab 版本

根据本仓库配置，**navilab 声明支持以下 Isaac Sim 版本**（见 `source/navilab/setup.py`）：

| Isaac Sim 版本 | 说明 |
|----------------|------|
| **4.5.0**      | 支持，需 Python 3.10 |
| **5.0.0**      | 支持 |
| **5.1.0**      | 支持，推荐（与当前 Isaac Lab main 一致） |

- **Isaac Lab**：本仓库中的 Isaac Lab 为 **v0.52.1** 风格（main 分支），官方说明支持 **Isaac Sim 4.5 / 5.0 / 5.1**。
- **Python**：Isaac Sim 5.x 需 **Python 3.11**；Isaac Sim 4.x 需 **Python 3.10**。

若云服务器上的 Isaac Sim 版本**不在** 4.5.0 / 5.0.0 / 5.1.0 中，可能出现 API 不兼容，需要按下面「移植办法」处理。

---

## 二、其他判断 Isaac Sim 版本的方法

在无法使用 `get_isaac_sim_version` 或需要多种 fallback 时，可用下面任一方式。

| 方法 | 说明 | 依赖 |
|------|------|------|
| **1. isaacsim.core.version** | 官方扩展，返回完整元组 | 需已启动 Isaac Sim 环境 |
| **2. SimulationContext.get_version()** | 仿真上下文实例方法，返回 (major, minor, patch) | 需已创建 SimulationContext |
| **3. 读 VERSION 文件** | 读安装目录下的 `VERSION` 文本文件 | 仅适用于手动/二进制安装，路径因安装方式而异 |
| **4. importlib.metadata** | 读已安装的 `isaacsim` 包版本 | 仅适用于 pip 安装的 Isaac Sim |

示例代码（按优先级尝试）：

```python
from packaging.version import Version

def get_isaac_sim_version_fallback():
    """多种方式获取 Isaac Sim 版本，返回 packaging.version.Version 或 None。"""
    # 1) Isaac Lab 封装（推荐）
    try:
        from isaaclab.utils.version import get_isaac_sim_version
        return get_isaac_sim_version()
    except ImportError:
        pass

    # 2) Isaac Sim 官方 API
    try:
        from isaacsim.core.version import get_version
        v = get_version()
        return Version(f"{v[2]}.{v[3]}.{v[4]}")
    except Exception:
        pass

    # 3) 已有 SimulationContext 时
    try:
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        if sim is not None:
            maj, min_, patch = sim.get_version()
            return Version(f"{maj}.{min_}.{patch}")
    except Exception:
        pass

    # 4) 读 VERSION 文件（Isaac Sim 安装根目录下的 VERSION）
    try:
        import isaacsim
        import os
        version_path = os.path.abspath(os.path.join(os.path.dirname(isaacsim.__file__), "../../VERSION"))
        if os.path.isfile(version_path):
            with open(version_path) as f:
                ver = f.readline().strip()
                return Version(ver.split("-")[0])  # 去掉可能的 -pre 等后缀
    except Exception:
        pass

    # 5) pip 安装的 isaacsim 包
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version("isaacsim")
        return Version(ver.split("-")[0])
    except Exception:
        pass

    return None
```

仅做 major 判断时，也可用 Isaac Lab 自带的 `AppLauncher.is_isaac_sim_version_4_5()`（需在 AppLauncher 已创建后调用），用于判断是否为 4.5.x。

---

## 三、移植办法

### 方案 A：让云服务器版本与本地一致（推荐）

1. **查云服务器上的 Isaac Sim 版本**  
   在 Isaac Sim 的 Python 环境中执行：
   ```python
   from isaacsim.core.version import get_version
   v = get_version()
   print(f"{v[2]}.{v[3]}.{v[4]}")  # major.minor.patch
   ```
   或使用 Isaac Lab 提供的接口（需较新 Isaac Lab）：
   ```python
   from isaaclab.utils.version import get_isaac_sim_version
   print(get_isaac_sim_version())
   ```
   若报错 `cannot import name get_isaac_sim_version`，说明当前 Isaac Lab 较旧，请用上面的 `isaacsim.core.version.get_version` 方式。

2. **若服务器版本不在 4.5/5.0/5.1**：
   - 在云服务器上安装 **Isaac Sim 4.5.0、5.0.0 或 5.1.0** 之一（与本地一致或选 5.1.0 最省心），然后在该环境中安装本仓库的 Isaac Lab + navilab。
   - 或：在本地改用与云服务器**相同版本**的 Isaac Sim + Isaac Lab，再迁代码（见方案 B）。

这样无需改 navilab 代码，兼容性由版本一致来保证。

---

### 方案 B：在代码中做版本兼容（服务器版本固定且无法改时）

若云服务器只能使用某一版本（例如 4.2.0 或 6.0.0），可以：

1. **确认该版本与 4.5/5.0/5.1 的 API 差异**  
   查阅 [Isaac Sim 发布说明](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html) 和 Isaac Lab 的 `CHANGELOG`，看是否有废弃/改名接口。

2. **在 navilab 中用版本分支**（若确有 API 差异）：
   ```python
   from isaaclab.utils.version import get_isaac_sim_version
   from packaging.version import Version

   ver = get_isaac_sim_version()
   if ver >= Version("5.0.0"):
       # 5.x 写法
   else:
       # 4.x 写法
   ```
   仅对确实不一致的调用处做分支，避免大改。

3. **更新 `setup.py` 的 classifiers**  
   若你明确支持了新版本（如 6.0.0），在 `source/navilab/setup.py` 里增加对应条目，例如：
   ```python
   "Isaac Sim :: 6.0.0",
   ```
   并注明在 README 或本文件中。

---

### 方案 C：用 Docker 固定 Isaac Sim 版本（适合训练/CI）

通过 Docker 在云上使用**与本地相同**的 Isaac Sim 版本，避免环境漂移。

1. **使用 Isaac Lab 官方 Docker 流程**  
   参考仓库内：
   - `IsaacLab/docker/`：基础镜像与编排
   - `IsaacLab/docker/.env.base`：可设置 `ISAACSIM_VERSION=5.1.0`（或 4.5.0 / 5.0.0）
   - `IsaacLab/tools/template/templates/external/docker/`：扩展（含 navilab）的 Docker 示例

2. **为 navilab 做一份 Docker 配置**（可选）  
   - 复制 `IsaacLab/tools/template/templates/external/docker/` 到 navilab 仓库（如 `navilab/docker/`）。
   - 在 `.env` 或 `.env.base` 中设置：
     - `ISAACSIM_VERSION=5.1.0`（或你本地使用的版本）
     - 以及 Isaac Lab 路径、扩展路径等（与模板一致）。
   - 构建镜像后，在容器内安装 navilab：  
     `pip install -e source/navilab`，再运行训练脚本。

这样云上环境与本地完全一致，便于复现和调试。

---

### 方案 D：仅文档化要求，由运维保证版本

在 README 或本文件中写明：

- **支持的 Isaac Sim 版本**：4.5.0、5.0.0、5.1.0。
- **推荐的 Isaac Lab 版本**：与当前仓库中 Isaac Lab 一致（main / v0.52.x 对应 4.5/5.0/5.1）。
- **Python**：3.10（Isaac Sim 4.x）或 3.11（Isaac Sim 5.x）。

由负责云环境的人在对应机器上安装上述组合，避免使用未声明的版本。

---

## 四、建议流程总结

1. **在云服务器上查版本**（见方案 A 的代码）。
2. **若版本是 4.5.0 / 5.0.0 / 5.1.0**：直接在该环境安装本仓库的 Isaac Lab + navilab，无需改代码。
3. **若版本不是上述三者**：  
   - 优先：在云上安装 5.1.0（或与本地一致的 4.5/5.0/5.1）（方案 A）。  
   - 若无法改云上 Isaac Sim 版本：再做版本分支或适配（方案 B），并更新 `setup.py` 与本文档。  
   - 若希望环境可复现：用 Docker 固定 Isaac Sim 版本（方案 C）。

---

## 五、参考链接

- [Isaac Lab 安装与版本对应](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)  
- [Isaac Sim 发布说明](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html)  
- 本仓库 `IsaacLab/README.md` 中的 “Isaac Sim Version Dependency” 表格  
