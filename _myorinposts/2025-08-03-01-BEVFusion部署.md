---
title: '2025-08-02-01-关于vic的使用'
date: 2025-08-02
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true

permalink: /myorinposts/2025/08/2025-08-02-01-关于vic的使用/
tags:
  - sdk

---


本文记录使用VIC 进行YUVx 到NV12的转换。   
注：至于其他resize,crop等操作可参考nv sample。   




```cpp

NvMedia2DComposeParameters params;
NvMedia2DGetComposeParameters(handle, &params);
NvMedia2DSetSrcNvSciBufObj(handle, params, 0, srcBuf);
NvMedia2DSetDstNvSciBufObj(handle, params, dstBuf);
NvMedia2DCompose(handle, params, NULL);
```

https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-linux-sdk/api_reference/group__x__nvmedia__2d__api.html#gae58a626650f2e7679c9be4904c22553b

