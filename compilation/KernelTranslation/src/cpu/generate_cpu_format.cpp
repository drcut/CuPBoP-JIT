#include "generate_cpu_format.h"
#include "debug.hpp"
#include "tool.h"
#include "llvm/Support/Host.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

// set TargetTriple and DataLayout same as the host CPU
void set_meta_data(llvm::Module *M) {
  M->setTargetTriple(llvm::sys::getProcessTriple());
  // use the default DataLayout
  M->setDataLayout("");
}

// as pthread only accept a single void* for input
// we have to decode this input inside the kernel
void generate_kernel_launch_wrapper(llvm::Module *M,
                                    llvm::Module *host_module) {

  // This is part of block-size invariant analysis.
  // We scan the host module to get all possible block sizes.
  std::set<Dim3SizeConfig> possible_block_size_list =
      get_possible_grid_or_block_size(host_module, /*getBlockSize=*/true);
  if (possible_block_size_list.size() != 0) {
    printf("possible block sizes: \n");
    for (auto block_size : possible_block_size_list) {
      printf("x: %d y: %d z: %d\n", block_size._x, block_size._y,
             block_size._z);
    }
    printf("\n");
  }

  std::set<llvm::Function *> need_remove;

  llvm::Type *Int32T = Type::getInt32Ty(M->getContext());
  llvm::Type *Int8T = Type::getInt8Ty(M->getContext());

  llvm::FunctionType *LauncherFuncT = FunctionType::get(
      Type::getVoidTy(M->getContext()), {PointerType::get(Int8T, 0)}, false);

  // generate Wrapper Function type
  // now we only support a single int32*
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    Function *F = &(*i);
    if (!isKernelFunction(M, F))
      continue;
    auto func_name = F->getName().str();
    // filter out _Z24 and other mangled prefix
    for (int pos = 2; pos < func_name.length(); pos++) {
      if (func_name[pos] >= '0' && func_name[pos] <= '9')
        continue;
      func_name = func_name.substr(pos);
      break;
    }
    llvm::IRBuilder<> Builder(M->getContext());

    FunctionCallee fc =
        M->getOrInsertFunction(func_name + "_wrapper", LauncherFuncT);
    Function *WorkGroup = dyn_cast<Function>(fc.getCallee());

    BasicBlock *Block = BasicBlock::Create(M->getContext(), "", WorkGroup);
    Builder.SetInsertPoint(Block);

    // WorkGroup has only a single input
    Function::arg_iterator ai = WorkGroup->arg_begin();

    SmallVector<Value *, 8> Arguments;
    Value *input_arg = &*ai;
    // convert to int**
    input_arg = Builder.CreateBitOrPointerCast(
        input_arg, PointerType::get(PointerType::get(Int32T, 0), 0));

    size_t idx = 0;
    // replace original arguments with the unpacked values
    // for example, for a function f(int* a, char* b),
    // we will generate a function f_wrapper(int** input)
    // and replace the original arguments with the unpacked values
    // e.g., a = (int*)input[0], b = (char*)input[1]
    for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
         ii != ee; ++ii) {
      Type *ArgType = ii->getType();
      // calculate addr
      Value *GEP = createGEP(Builder, input_arg, ConstantInt::get(Int32T, idx));
      // load corresponding int*
      GEP = createLoad(Builder, GEP);
      // bitcast
      GEP = Builder.CreateBitOrPointerCast(GEP, PointerType::get(ArgType, 0));
      Value *Arg = createLoad(Builder, GEP);
      Arguments.push_back(Arg);
      ++idx;
    }
    BasicBlock *exit_block = BasicBlock::Create(M->getContext(), "", WorkGroup);
    {
      llvm::IRBuilder<> Builder3(M->getContext());
      Builder3.SetInsertPoint(exit_block);
      Builder3.CreateRetVoid();
    }
    auto block_size_global = M->getGlobalVariable("block_size");
    auto loaded_block_size = Builder.CreateLoad(
        block_size_global->getType()->getElementType(), block_size_global);
    BasicBlock *default_call_block =
        BasicBlock::Create(M->getContext(), "default_block_size", WorkGroup);
    auto default_call_inst = llvm::CallInst::Create(
        F, ArrayRef<llvm::Value *>(Arguments), "", default_call_block);
    llvm::BranchInst::Create(exit_block, default_call_block);
    auto switchInst =
        Builder.CreateSwitch(loaded_block_size, default_call_block);

    for (auto block_size_config : possible_block_size_list) {
      // Clone a new function
      ValueToValueMapTy EmptyMap;
      Function *Clone = CloneFunction(F, EmptyMap);
      Clone->setName(F->getName() + "_block_size_" +
                     block_size_config.toString());
      // replace all reference to block_size to constants
      auto replace_global_variable_with_constant =
          [&](llvm::GlobalVariable *global_variable, int constant) {
            ConstantInt *constant_int =
                dyn_cast<ConstantInt>(ConstantInt::get(Int32T, constant, true));
            std::vector<llvm::Value *> Users(global_variable->user_begin(),
                                             global_variable->user_end());
            for (auto *U : Users) {
              if (auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(U))
                if (loadInst->getParent()->getParent() == Clone) {
                  loadInst->replaceAllUsesWith(constant_int);
                }
            }
          };

      int total_block_size =
          block_size_config._x * block_size_config._y * block_size_config._z;
      replace_global_variable_with_constant(M->getGlobalVariable("block_size"),
                                            total_block_size);
      replace_global_variable_with_constant(
          M->getGlobalVariable("block_size_x"), block_size_config._x);
      replace_global_variable_with_constant(
          M->getGlobalVariable("block_size_y"), block_size_config._y);
      replace_global_variable_with_constant(
          M->getGlobalVariable("block_size_z"), block_size_config._z);
      BasicBlock *possible_call_block = BasicBlock::Create(
          M->getContext(), "block_size_" + block_size_config.toString(),
          WorkGroup);
      auto call_inst = llvm::CallInst::Create(
          Clone, ArrayRef<llvm::Value *>(Arguments), "", possible_call_block);
      auto branch_inst =
          llvm::BranchInst::Create(exit_block, possible_call_block);
      switchInst->addCase(dyn_cast<ConstantInt>(
                              ConstantInt::get(Int32T, total_block_size, true)),
                          possible_call_block);
    }
  }
  for (auto f : need_remove) {
    f->dropAllReferences();
    f->eraseFromParent();
  }
}

// after flat/hierarchical collapsing, the barrier instructions are useless
void remove_barrier(llvm::Module *M) {
  std::vector<Instruction *> need_remove;
  for (auto F = M->begin(); F != M->end(); ++F)
    for (auto BB = F->begin(); BB != F->end(); ++BB) {
      for (auto Inst = BB->begin(); Inst != BB->end(); Inst++) {
        if (auto Call = dyn_cast<CallInst>(Inst)) {
          if (Call->isInlineAsm())
            continue;
          auto func_name = Call->getCalledFunction()->getName().str();
          if (func_name == "llvm.nvvm.bar.warp.sync" ||
              func_name == "llvm.nvvm.barrier0" ||
              func_name == "llvm.nvvm.barrier.sync") {
            need_remove.push_back(Call);
          }
        }
      }
    }
  for (auto inst : need_remove) {
    inst->eraseFromParent();
  }
}

void remove_useless_var(llvm::Module *M) {
  M->getGlobalVariable("intra_warp_index")->eraseFromParent();
  M->getGlobalVariable("inter_warp_index")->eraseFromParent();
}

void generate_cpu_format(llvm::Module *kernel, llvm::Module *host) {
  DEBUG_INFO("generate cpu format\n");
  // change metadata
  set_meta_data(kernel);
  // decode argument
  generate_kernel_launch_wrapper(kernel, host);
  // remove barrier
  remove_barrier(kernel);
  // remove useless func/variable
  remove_useless_var(kernel);
}
