#include "replace_configuration.h"
#include "debug.hpp"
#include "tool.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <assert.h>
#include <iostream>
#include <set>

using namespace llvm;

namespace {

void replace_global_variable(llvm::GlobalVariable *gv, llvm::Value *v) {
  if (!gv)
    return;
  std::vector<llvm::Value *> Users(gv->user_begin(), gv->user_end());
  for (auto *U : Users) {
    if (auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(U)) {
      loadInst->replaceAllUsesWith(v);
    } else {
      // as we only replace block_size global variables, all users should be
      // loadInst
      exit(1);
    }
  }
}

struct ReplaceRuntimeConfiguration : public ModulePass {
  static char ID;
  size_t grid_x;
  size_t grid_y;
  size_t grid_z;
  size_t block_x;
  size_t block_y;
  size_t block_z;
  ReplaceRuntimeConfiguration(size_t _grid_x, size_t _grid_y, size_t _grid_z,
                              size_t _block_x, size_t _block_y, size_t _block_z)
      : ModulePass(ID), grid_x(_grid_x), grid_y(_grid_y), grid_z(_grid_z),
        block_x(_block_x), block_y(_block_y), block_z(_block_z) {}

  bool runOnModule(Module &M) override {
    llvm::Type *I32 = llvm::Type::getInt32Ty(M.getContext());

    auto grid_x_constant = llvm::ConstantInt::get(I32, grid_x, true);
    auto grid_y_constant = llvm::ConstantInt::get(I32, grid_y, true);
    auto grid_z_constant = llvm::ConstantInt::get(I32, grid_z, true);

    auto block_x_constant = llvm::ConstantInt::get(I32, block_x, true);
    auto block_y_constant = llvm::ConstantInt::get(I32, block_y, true);
    auto block_z_constant = llvm::ConstantInt::get(I32, block_z, true);
    auto block_size_constant =
        llvm::ConstantInt::get(I32, block_x * block_y * block_z, true);
    // replace the global variable
    replace_global_variable(M.getGlobalVariable("grid_size_x"),
                            grid_x_constant);
    replace_global_variable(M.getGlobalVariable("grid_size_y"),
                            grid_y_constant);
    replace_global_variable(M.getGlobalVariable("grid_size_z"),
                            grid_z_constant);
    replace_global_variable(M.getGlobalVariable("block_size"),
                            block_size_constant);
    replace_global_variable(M.getGlobalVariable("block_size_x"),
                            block_x_constant);
    replace_global_variable(M.getGlobalVariable("block_size_y"),
                            block_y_constant);
    replace_global_variable(M.getGlobalVariable("block_size_z"),
                            block_z_constant);
    replace_global_variable(M.getGlobalVariable("block_size"),
                            block_size_constant);
  }
};
} // namespace

char ReplaceRuntimeConfiguration::ID = 0;
/*
replace block_size with constant
*/
void replace_grid_block_size_global_variable(llvm::Module *M, size_t grid_x,
                                             size_t grid_y, size_t grid_z,
                                             size_t block_x, size_t block_y,
                                             size_t block_z) {
  DEBUG_INFO("replace block size variables to constants\n");
  llvm::legacy::PassManager Passes;
  Passes.add(new ReplaceRuntimeConfiguration(grid_x, grid_y, grid_z, block_x,
                                             block_y, block_z));
  Passes.run(*M);
}
