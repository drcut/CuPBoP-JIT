#!/bin/bash
set -e

kernelTranslator hist-cuda-nvptx64-nvidia-cuda-sm_50.bc hist-host-x86_64-unknown-linux-gnu.bc kernel.bc
hostTranslator hist-host-x86_64-unknown-linux-gnu.bc host.bc

llc --relocation-model=pic --filetype=obj  kernel.bc
llc --relocation-model=pic --filetype=obj  host.bc

g++ -o hist -fPIC -no-pie -L$CuPBoP_BUILD_PATH/runtime \
  -L$CuPBoP_BUILD_PATH/runtime/threadPool \
  host.o kernel.o -lpthread -lc -lCPUruntime -lthreadPool

./hist
