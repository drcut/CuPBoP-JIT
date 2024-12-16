#include "generate_cpu_format.h"
#include "debug.hpp"
#include "tool.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <set>
using namespace llvm;

// set TargetTriple and DataLayout same as the host CPU
void set_meta_data(llvm::Module *M) {
  M->setTargetTriple(llvm::sys::getProcessTriple());
  // use the default DataLayout
  M->setDataLayout("");
}

// as pthread only accept a single void* for input
// we have to decode this input inside the kernel
void decode_input(llvm::Module *M) {
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
      Value *GEP = Builder.CreateGEP(PointerType::get(Int32T, 0), input_arg,
                                     ConstantInt::get(Int32T, idx));
      // load corresponding int*
      GEP = Builder.CreateLoad(PointerType::get(Int32T, 0), GEP);
      // bitcast
      GEP = Builder.CreateBitOrPointerCast(GEP, PointerType::get(ArgType, 0));
      Value *Arg = Builder.CreateLoad(ArgType, GEP);
      Arguments.push_back(Arg);
      ++idx;
    }
    Builder.CreateCall(F, ArrayRef<llvm::Value *>(Arguments));
    Builder.CreateRetVoid();
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

// Since Triton cannot support extern variables, and we need to
// make kernel as a shared library. We need to replace the extern
// variables by arguments
void replace_global_variables(Module *M) {
  std::vector<std::string> globalVariablesToReplace = {
      "block_size",    "block_size_x",  "block_size_y",         "block_size_z",
      "grid_size_x",   "grid_size_y",   "grid_size_z",          "block_index_x",
      "block_index_y", "block_index_z", "dynamic_shared_memory"};
  std::map<std::string, GlobalVariable *> globalsMap;
  for (auto &GVName : globalVariablesToReplace) {
    GlobalVariable *GV = M->getGlobalVariable(GVName);
    if (GV) {
      globalsMap[GVName] = GV;
    }
  }

  std::set<Function *> need_wrapper;
  for (Function &F : *M) {
    if (!isKernelFunction(M, &F))
      continue;
    need_wrapper.insert(&F);
  }
  for (auto &F : need_wrapper) {
    // Clone the function, with a new signature (more arguments)
    std::vector<Type *> paramTypes;
    for (auto &Arg : F->args()) {
      paramTypes.push_back(Arg.getType());
    }
    // Add types of the global variables
    for (auto global_name : globalVariablesToReplace) {
      GlobalVariable *GV = M->getGlobalVariable(global_name);
      paramTypes.push_back(GV->getValueType());
    }

    // Create new function
    FunctionType *newFuncType =
        FunctionType::get(F->getReturnType(), paramTypes, F->isVarArg());
    Function *newFunc =
        Function::Create(newFuncType, F->getLinkage(),
                         F->getName() + "_wrapper", F->getParent());
    newFunc->copyAttributesFrom(F);
    newFunc->setCallingConv(F->getCallingConv());

    ValueToValueMapTy VMap;
    ValueToValueMapTy argMap;
    auto newArgIt = newFunc->arg_begin();
    for (auto &Arg : F->args()) {
      newArgIt->setName(Arg.getName());
      VMap[&Arg] = &*newArgIt;
      ++newArgIt;
    }
    for (auto &GVName : globalVariablesToReplace) {
      newArgIt->setName(GVName);
      GlobalVariable *GV = globalsMap[GVName];
      argMap[GV] = &*newArgIt;
      ++newArgIt;
    }
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(newFunc, F, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);
    F->replaceAllUsesWith(newFunc);
    F->eraseFromParent();

    // Replace the global variable with the argument
    std::set<LoadInst *> need_replace;
    for (Instruction &I : instructions(*newFunc)) {
      if (llvm::LoadInst *loadInst = dyn_cast<llvm::LoadInst>(&I)) {
        if (GlobalVariable *GV =
                dyn_cast<GlobalVariable>(loadInst->getOperand(0))) {
          if (globalsMap.find(GV->getName().str()) != globalsMap.end())
            need_replace.insert(loadInst);
        }
      }
    }
    for (auto inst : need_replace) {
      auto newArgIt = argMap[cast<GlobalVariable>(inst->getOperand(0))];
      inst->replaceAllUsesWith(&*newArgIt);
      inst->eraseFromParent();
    }
  }
  // Erase the global variables
  for (auto &GVName : globalVariablesToReplace) {
    GlobalVariable *GV = M->getGlobalVariable(GVName);
    if (GV) {
      GV->eraseFromParent();
    }
  }
}

void generate_cpu_format(llvm::Module *M) {
  DEBUG_INFO("generate cpu format\n");
  // Reset metadata
  set_meta_data(M);
  // Replace global variables with arguments
  replace_global_variables(M);
  // Remove barrier
  remove_barrier(M);
  // Remove useless func/variable
  remove_useless_var(M);
}
