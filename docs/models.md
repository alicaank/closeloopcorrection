# Model adapters

These are pinned local integrations, not a claim that every checkpoint or new guard has passed end-to-end GPU validation. `native_callback` means the reusable API requires the model-owned factor; it will not guess one. Legacy launchers retain their original code paths.

| Model / upstream | Scale strategy in core | Native camera basis | State / limitations |
|---|---|---|---|
| [Matrix-Game 3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) (`matrixgame`) | `metric` | c2w / opencv | Metric metadata drives the adapter; native action encoding remains model-owned. Training corpus unresolved. |
| [Lyra-2](https://github.com/nv-tlabs/lyra) (`lyra`) | `anchor_depth` | w2c / opencv | Final native DA3/MoGe anchor depth; 96 history padding frames are not generated time. v3 quality gate awaits GPU validation. |
| [Spatia](https://github.com/ZhaoJingjing713/Spatia) (`spatia`) | `anchor_depth` | w2c / opencv | Anchor MapAnything depth; w2c translations in native units. v3 view batching awaits CUDA parity. |
| [VMem](https://github.com/runjiali-rl/vmem) (`vmem`) | `native_callback` | c2w / opengl | OpenGL cameras and native per-window translation normalization; obtain the factor from the released pipeline, do not freeze it globally. |
| [Astronex](https://github.com/Astronex-Robotics/Astronex-World) (`astronex`) | `native_callback` | c2w / opencv | Latent frame j corresponds to RGB frame 4j; metric/native normalization needs an explicit callback. |
| [LingBot v2](https://github.com/robbyant/lingbot-world-v2) (`lingbot`) | `native_callback` | c2w / opencv | Relative camera chain divided by the frozen native path normalizer. Preserve the native relative/Plucker encoding. |
| [DreamX-World](https://github.com/AMAP-ML/DreamX-World) (`dreamx`) | `native_callback` | c2w / opencv | Camera indices are [0,1,5,...]; decoded feedback is delayed by three frames. Keep its adapter-specific alignment. |
| [WorldWarp](https://github.com/HyoKong/WorldWarp) (`worldwarp`) | `anchor_depth` | c2w / opencv | Initial TTT3R depth fit plus a two-chunk continuation gate; later chunks are unsupported by that gate. GPU validation pending. |
| [SANA-WM](https://github.com/NVlabs/Sana) (`sana`) | `metric` | c2w / opencv | Metric sidecar conversion, 24-frame chunks; preserve cached past camera conditioning. GPU validation pending. |
| [Memorizon](https://github.com/TingtingLiao/memorizon) (`memorizon`) | `metric_div4` | c2w / opencv | Recorded metric translation /4 mapping; projection audit pending. Historical launcher lacks native/sham controls. |
| [WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) (`worldplay`) | `native_callback` | w2c / opencv | Native pose-unit contract unresolved. Explicit calibrated factor required; historical launcher lacks native/sham controls. |

## Source pins

- **Matrix-Game 3.5**: `fbf7def0693ae14f745ba35bf2a26215d4ef991d`
- **Lyra-2**: `9fffc9adc37004091ecf26ef03abfb3abdf4d59a`
- **Spatia**: `f75b10c9bb5f6b0cd779ec8c41ab33dd8382dba3`
- **VMem**: `39291e4f272f6b4f270691d930926ab5930f942e`
- **Astronex**: `27584bf1a89da01ef35a03b48d83bc818d2adf2b`
- **LingBot v2**: `1895d300d8ac936401689b26389f51cbd36530eb`
- **DreamX-World**: `a1f4c6e5e45600718e5236955f2e0702e53fc275`
- **WorldWarp**: `0396b801c278546d01a6618a18e5a5c2a154c0a1`
- **SANA-WM**: `25adaaf02ad501c27be2c376b6b49b349cbcbe52`
- **Memorizon**: `432a49e01c19be8fbd60eb37427ed07dfcc2db16`
- **WorldPlay**: `1588e1336e842b03b0a7860c654ebd7c46bb065e`

SolarWM and WorldCrafter are not listed as supported because no working local adapter was included. The source inventory is not a verified checkpoint training-corpus or held-out-split mapping.
