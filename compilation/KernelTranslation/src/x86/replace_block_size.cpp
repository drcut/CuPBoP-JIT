#include "replace_block_size.h"
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
      printf("replace\n");
      loadInst->dump();
      v->dump();
      loadInst->replaceAllUsesWith(v);
      printf("replace done\n");
    } else {
      // as we only replace block_size global variables, all users should be
      // loadInst
      exit(1);
    }
  }
  printf("get here\n");
  // gv->dropAllReferences();
  // gv->removeFromParent();
}

struct ReplaceBlockSize : public ModulePass {
  static char ID;
  size_t block_x;
  size_t block_y;
  size_t block_z;
  ReplaceBlockSize(size_t _block_x, size_t _block_y = 1, size_t _block_z = 1)
      : ModulePass(ID), block_x(_block_x), block_y(_block_y),
        block_z(_block_z) {}

  bool runOnModule(Module &M) override {
    llvm::Type *I32 = llvm::Type::getInt32Ty(M.getContext());

    auto block_x_constant = llvm::ConstantInt::get(I32, block_x, true);
    auto block_y_constant = llvm::ConstantInt::get(I32, block_y, true);
    auto block_z_constant = llvm::ConstantInt::get(I32, block_z, true);
    auto block_size_constant =
        llvm::ConstantInt::get(I32, block_x * block_y * block_z, true);
    // replace the global variable
    printf("p1\n");
    replace_global_variable(M.getGlobalVariable("block_size_x"),
                            block_x_constant);
    printf("p1\n");
    replace_global_variable(M.getGlobalVariable("block_size_y"),
                            block_y_constant);
    printf("p1\n");
    replace_global_variable(M.getGlobalVariable("block_size_z"),
                            block_z_constant);
    printf("p1\n");
    replace_global_variable(M.getGlobalVariable("block_size"),
                            block_size_constant);
    printf("dump\n");
  }
};
} // namespace

char ReplaceBlockSize::ID = 0;
/*
replace block_size with constant
*/
void replace_block_size_global_variable(llvm::Module *M, size_t block_x,
                                        size_t block_y, size_t block_z) {
  DEBUG_INFO("replace block size variables to constants\n");
  llvm::legacy::PassManager Passes;
  Passes.add(new ReplaceBlockSize(block_x, block_y, block_z));
  Passes.run(*M);
}
