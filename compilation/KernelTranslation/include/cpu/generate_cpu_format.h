#ifndef __NVVM2CPU_GENERATE_CPU_FORMAT__
#define __NVVM2CPU_GENERATE_CPU_FORMAT__

#include "llvm/IR/Module.h"

void generate_cpu_format(llvm::Module *kernel, llvm::Module *host);

void set_meta_data(llvm::Module *M);

#endif
