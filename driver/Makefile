# SPDX-License-Identifier: GPL-2.0
# Copyright (C) 2024 Arm Technology (China) Co. Ltd.
# Modified for Orange Pi 6 Plus (CIX CD8180 / SKY1 / Zhouyi V3) on Armbian

ifneq ($(KERNELRELEASE),)

ccflags-y += -DCONFIG_SKY1 \
             -DCONFIG_ARMCHINA_NPU_ARCH_V3 \
             -DBUILD_ZHOUYI_V3 \
             -DKMD_VERSION=\"5.11.0\" \
             -I$(src)/armchina-npu/ \
             -I$(src)/armchina-npu/include \
             -I$(src)/armchina-npu/zhouyi

obj-m := aipu.o
aipu-y := armchina-npu/sky1/sky1.o \
          armchina-npu/aipu.o \
          armchina-npu/aipu_common.o \
          armchina-npu/aipu_io.o \
          armchina-npu/aipu_irq.o \
          armchina-npu/aipu_job_manager.o \
          armchina-npu/aipu_mm.o \
          armchina-npu/aipu_dma_buf.o \
          armchina-npu/aipu_priv.o \
          armchina-npu/aipu_tcb.o \
          armchina-npu/zhouyi/zhouyi.o \
          armchina-npu/zhouyi/v3.o \
          armchina-npu/zhouyi/v3_priv.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

endif
