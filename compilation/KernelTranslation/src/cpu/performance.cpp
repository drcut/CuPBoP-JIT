#include "performance.h"
#include "debug.hpp"
#include "tool.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

void performance_optimization(llvm::Module *M) {
  DEBUG_INFO("performance optimization\n");
  for (auto F = M->begin(); F != M->end(); F++) {
    for (auto I = F->arg_begin(); I != F->arg_end(); ++I) {
      if (I->getType()->isPointerTy()) {
        I->addAttr(llvm::Attribute::NoAlias);
      }
    }
  }

  llvm::PassBuilder PassBuilder;
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  PassBuilder.registerModuleAnalyses(MAM);
  PassBuilder.registerCGSCCAnalyses(CGAM);
  PassBuilder.registerFunctionAnalyses(FAM);
  PassBuilder.registerLoopAnalyses(LAM);
  PassBuilder.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM;
  llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O3;
  MPM = PassBuilder.buildPerModuleDefaultPipeline(OptLevel);
  MPM.run(*M, MAM);
}
