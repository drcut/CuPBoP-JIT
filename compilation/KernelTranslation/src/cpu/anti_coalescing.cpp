#include "anti_coalescing.h"
#include "debug.hpp"
#include "performance.h"
#include "tool.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

// Check whether the value of inst is linear with blockDim
bool linear_with_blockDim(llvm::Instruction *inst,
                          std::set<llvm::Instruction *> visited) {

  bool result = false;
  if (llvm::CallBase *callInst = llvm::dyn_cast<llvm::CallBase>(inst)) {
    if (Function *calledFunction = callInst->getCalledFunction()) {
      if (calledFunction->getName().startswith(
              "llvm.nvvm.read.ptx.sreg.ntid")) {
        result = true;
      }
    } else {
      result = false;
    }
  } else if (auto binOp = llvm::dyn_cast<llvm::BinaryOperator>(inst)) {
    if (auto lhs = dyn_cast<llvm::Instruction>(binOp->getOperand(0)))
      if (visited.find(lhs) == visited.end()) {
        visited.insert(lhs);
        result |= linear_with_blockDim(lhs, visited);
      }

    if (auto rhs = dyn_cast<llvm::Instruction>(binOp->getOperand(1)))
      if (visited.find(rhs) == visited.end()) {
        visited.insert(rhs);
        result |= linear_with_blockDim(rhs, visited);
      }
  } else if (auto loadInst = llvm::dyn_cast<llvm::LoadInst>(inst)) {
    // find all store that related to this load
    auto address = loadInst->getOperand(0);
    bool all_linear_with_blockDim = true;
    for (auto U : address->users()) {
      if (auto store = dyn_cast<StoreInst>(U)) {
        all_linear_with_blockDim &= linear_with_blockDim(store, visited);
      } else if (!isa<AllocaInst>(U) && U != loadInst) {
        all_linear_with_blockDim = false;
      }
    }
    result = all_linear_with_blockDim;
  } else if (auto storeInst = llvm::dyn_cast<llvm::StoreInst>(inst)) {
    if (isa<llvm::Instruction>(storeInst->getOperand(0)))
      result = linear_with_blockDim(
          dyn_cast<llvm::Instruction>(storeInst->getOperand(0)), visited);
  }
  return result;
}
// Check whether "des" is linear related with "src"
bool linear_related(llvm::Instruction *src, llvm::Instruction *des) {
  if (src == des)
    return true;
  if (auto cast = dyn_cast<CastInst>(des)) {
    if (auto cast_var = dyn_cast<llvm::Instruction>(cast->getOperand(0)))
      return linear_related(src, cast_var);
  } else if (auto load = dyn_cast<LoadInst>(des)) {
    if (auto load_var = dyn_cast<llvm::Instruction>(load->getOperand(0)))
      return linear_related(src, load_var);
  }
  return false;
}
/*
 * Check whether the loop contains global memory coalescing optimizations
 * We identify the memory coalescing according to that:
 * 1) the loop iteration variable is linearly related with blockDim;
 * 2) there are memory access, where the index is: induction_variable +
 * threadIdx and the index is the -1 dimension;
 */
bool loop_contains_global_memory_coalescing(llvm::Loop *L) {
  printf("check loop with header: %s\n",
         L->getHeader()->getName().str().c_str());
  // find iteration variable
  auto loop_latch = L->getLoopLatch();
  auto F = loop_latch->getParent();
  if (!loop_latch) {
    return false;
  }
  llvm::Instruction *iteration_var = NULL;
  llvm::Instruction *inc_inst = NULL;
  for (BasicBlock::reverse_iterator i = loop_latch->rbegin(),
                                    e = loop_latch->rend();
       i != e; ++i) {
    if (auto Store = dyn_cast<llvm::StoreInst>(&*i)) {
      if (isa<llvm::Instruction>(Store->getOperand(1))) {
        iteration_var = dyn_cast<llvm::Instruction>(Store->getOperand(1));
        inc_inst = dyn_cast<llvm::Instruction>(Store->getOperand(0));

        break;
      }
    }
  }
  if (!iteration_var) {
    // cannot find iteration variable
    exit(1);
  }

  // check whether the stride is a linear function of blockDim
  std::set<llvm::Instruction *> visited;
  if (!linear_with_blockDim(inc_inst, visited)) {
    // the stride is not linear with blockDim
    printf("not linear with block\n");
    return false;
  }
  // check whether the loop contains GEP insturction, with iteration_var as the
  // last dimension
  for (Loop::block_iterator i = L->block_begin(), e = L->block_end(); i != e;
       ++i) {
    for (BasicBlock::iterator j = (*i)->begin(), e = (*i)->end(); j != e; ++j) {
      if (auto GEP = dyn_cast<GetElementPtrInst>(j)) {
        auto last_dim = GEP->getOperand(GEP->getNumIndices());
        if (auto last_dim_var = dyn_cast<llvm::Instruction>(last_dim)) {
          if (linear_related(iteration_var, last_dim_var)) {
            printf("Find global memory coalescing\n");
            return true;
          }
        }
      }
    }
  }
  return false;
}

struct AntiMemCoalescingTransformation : public llvm::FunctionPass {

public:
  static char ID;

  AntiMemCoalescingTransformation() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredTransitive<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) {
    auto M = F.getParent();
    if (!isKernelFunction(M, &F))
      return 0;
    // check whether this loop has barrier
    // if the loop contains barrier, we do not need to implement optimizations
    // for memory coalescing
    std::set<Loop *> mem_coalescing_loop;
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    SmallVector<Loop *, 8> LoopStack(LI.begin(), LI.end());
    for (auto L : LoopStack) {
      bool contains_barrier = 0;
      for (Loop::block_iterator i = L->block_begin(), e = L->block_end();
           i != e; ++i) {
        for (BasicBlock::iterator j = (*i)->begin(), e = (*i)->end(); j != e;
             ++j) {
          if (auto Call = dyn_cast<CallInst>(j)) {
            if (Call->isInlineAsm())
              continue;
            auto func_name = Call->getCalledFunction()->getName().str();
            if (func_name == "llvm.nvvm.barrier0" ||
                func_name == "llvm.nvvm.bar.warp.sync" ||
                func_name == "llvm.nvvm.barrier.sync") {
              contains_barrier = true;
              break;
            }
          }
        }
      }
      if (contains_barrier)
        continue;

      // check whether this loop contains global memory coalescing
      if (!loop_contains_global_memory_coalescing(L))
        continue;
      if (!L->getExitBlock())
        continue;
      mem_coalescing_loop.insert(L);
    }
    if (mem_coalescing_loop.size() == 0)
      return 0;
    // implement transformation
    for (auto L : mem_coalescing_loop) {
      LLVMContext &context = M->getContext();
      auto I8Ptr = llvm::Type::getInt8PtrTy(context);
      auto I8 = llvm::Type::getInt8Ty(context);
      IRBuilder<> builder(context);
      // create basic blocks
      BasicBlock *do_while_latch = BasicBlock::Create(
          context, "do_while_latch", L->getHeader()->getParent());
      BasicBlock *do_while_preheader = BasicBlock::Create(
          context, "do_while_preheader", L->getHeader()->getParent());
      BasicBlock *do_while_header = BasicBlock::Create(
          context, "do_while_header", L->getHeader()->getParent());
      // create do while latch
      builder.SetInsertPoint(do_while_latch);
      llvm::Instruction *has_activated_thread = createLoad(
          builder, M->getGlobalVariable("has_activated_thread_addr"));
      auto branch_var = builder.CreateICmpNE(has_activated_thread,
                                             ConstantInt::get(I8, 0, true));
      builder.CreateCondBr(branch_var, do_while_header, L->getExitBlock());
      // Part0: change the loop preheader, to jump to do while preheader
      auto preheader_br =
          dyn_cast<BranchInst>(L->getLoopPreheader()->getTerminator());
      assert(preheader_br->isUnconditional());
      preheader_br->setSuccessor(0, do_while_preheader);
      // Part1: do while preheader
      // create a variable thread_activated, to record whether a thread is
      // activated or not
      builder.SetInsertPoint(do_while_preheader);
      llvm::Instruction *thread_activated_addr =
          builder.CreateAlloca(I8, 0, "thread_activated_addr");
      builder.CreateStore(ConstantInt::get(I8, 1, true), thread_activated_addr);
      builder.CreateBr(do_while_header);
      // Part2: do while header
      // set has_activated_thread to false
      builder.SetInsertPoint(do_while_header);
      Instruction *last_inst = builder.CreateStore(
          ConstantInt::get(I8, 0),
          M->getGlobalVariable("has_activated_thread_addr"));
      // get condition ins the origial loop
      // (TODO): currently, we assume the loop header contains/calculates the
      // loop condition, and the cond is the instruction before condition
      llvm::BasicBlock *loop_body;
      llvm::BasicBlock *loop_cond = L->getHeader();
      for (auto b_iter = loop_cond->begin(); b_iter != loop_cond->end();
           ++b_iter) {
        Instruction *org_inst = dyn_cast<Instruction>(&*b_iter);
        if (isa<llvm::BranchInst>(org_inst)) {
          auto br = dyn_cast<llvm::BranchInst>(org_inst);
          loop_body = br->getSuccessor(0);
          break;
        }
        auto new_inst = org_inst->clone();
        new_inst->insertAfter(last_inst);
        org_inst->replaceAllUsesWith(new_inst);
        last_inst = new_inst;
      }
      if (!isa<llvm::CmpInst>(last_inst))
        return 0;
      CreateInterWarpBarrier(last_inst);
      // set thread_activated = thread_activated & (cond)
      auto thread_activated = new LoadInst(I8, thread_activated_addr,
                                           "thread_activated", do_while_header);
      last_inst = llvm::CastInst::CreateIntegerCast(last_inst, I8, false, "",
                                                    do_while_header);
      auto and_result = BinaryOperator::Create(
          Instruction::And, last_inst, thread_activated, "", do_while_header);
      new StoreInst(and_result, thread_activated_addr, do_while_header);
      // has_activated_thread |= and_result
      has_activated_thread =
          new LoadInst(I8, M->getGlobalVariable("has_activated_thread_addr"),
                       "has_activated_thread", do_while_header);
      auto or_result =
          BinaryOperator::Create(Instruction::Or, has_activated_thread,
                                 and_result, "", do_while_header);
      new StoreInst(or_result,
                    M->getGlobalVariable("has_activated_thread_addr"),
                    do_while_header);
      // create branch
      auto branch_res =
          new ICmpInst(*do_while_header, llvm::CmpInst::Predicate::ICMP_NE,
                       and_result, ConstantInt::get(I8, 0));
      llvm::BranchInst::Create(loop_body, do_while_latch, branch_res,
                               do_while_header);
      // Part3: replace original loop's latches target
      if (auto latch = L->getLoopLatch()) {
        auto t = dyn_cast<BranchInst>(latch->getTerminator());
        assert(t->isUnconditional());
        t->setSuccessor(0, do_while_latch);
      }
      // Part4: remove useless block: loop header, as it has been copied to
      // do_while_header
      DeleteDeadBlocks(loop_cond);
    }
    return 1;
  }
};

char AntiMemCoalescingTransformation::ID = 0;

namespace {
static RegisterPass<AntiMemCoalescingTransformation>
    insert_mem_coalescing_barrier(
        "anti-mem-coalescing-opt",
        "Apply anti-coalescing for global memory access");
} // namespace

void anti_global_mem_coalescing_optimization(llvm::Module *M) {
  DEBUG_INFO("anti global memory coalescing optimization\n");
  auto Registry = PassRegistry::getPassRegistry();

  llvm::legacy::PassManager Passes;

  std::vector<std::string> passes;
  passes.push_back("anti-mem-coalescing-opt");
  for (auto pass : passes) {
    const PassInfo *PIs = Registry->getPassInfo(StringRef(pass));
    if (PIs) {
      Pass *thispass = PIs->createPass();
      Passes.add(thispass);
    } else {
      assert(0 && "Pass not found\n");
    }
  }
  Passes.run(*M);
}
