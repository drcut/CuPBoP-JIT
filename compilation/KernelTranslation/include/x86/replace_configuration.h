#ifndef __NVVM2x86_REPLACE_CONFIGURATION__
#define __NVVM2x86_REPLACE_CONFIGURATION__

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

void replace_grid_block_size_global_variable(llvm::Module *M, size_t grid_x,
                                             size_t grid_y, size_t grid_z,
                                             size_t block_x, size_t block_y,
                                             size_t block_z);

#endif
