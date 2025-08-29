---
title: '关于nvscibuff中获取cuda指针的一次试验'
date: 2025-07-14
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true

permalink: /myorinposts/2025/07/关于nvscibuff中获取cuda指针的一次试验/
tags:
  - sdk

---

在 NVIDIA 平台上，像 CUDA、NvMedia、OpenGL 等库各自管理自己的内存资源，彼此之间缺乏直接的数据交换机制。NvSciBuf 的出现就是为了解决这个问题：

允许多个硬件引擎（如 GPU、ISP、DLA、VIC、NVENC、NVDEC 等）共享同一个缓冲区

满足所有访问者的约束条件，确保兼容性和安全性


## 核心工作流程

+ 创建属性列表（Attribute List） 每个访问者（如 CUDA 或 NvMedia）都需创建自己的属性列表，定义所需缓冲区的属性。

+ 设置属性   
  CUDA 侧需设置如 GpuId 等属性    
  NvMedia 侧需设置数据类型属性和通用属性

+ 协调属性列表（Reconcile） 将多个属性列表合并，生成一个满足所有访问者需求的对齐属性列表。

+ 分配缓冲区 使用协调后的属性列表分配缓冲区对象。

+ 共享缓冲区 将缓冲区对象分发给所有访问者，实现跨库共享