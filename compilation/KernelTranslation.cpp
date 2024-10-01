#include "debug.hpp"
#include "generate_cpu_format.h"
#include "handle_sync.h"
#include "init.h"
#include "insert_sync.h"
#include "insert_warp_loop.h"
#include "performance.h"
#include "tool.h"
#include "warp_func.h"
#include <assert.h>

using namespace llvm;

// to support constant memory variables, we need to convert information
// from kernelTranslator to HostTranslator, since HostTranslator knows nothing
// about the kernel functions, we need to write the information to a file
// by KernelTranslator and read it in HostTranslator
std::string PATH = "kernel_meta.log";

int main(int argc, char **argv) {
  assert(argc == 3 && "incorrect number of arguments\n");
  llvm::Module *program = LoadModuleFromFilr(argv[1]);

  std::ofstream fout;
  fout.open(PATH);

  DEBUG_INFO("CuPBoP pass: init_block\n");
  // inline __device__ functions, and create auxiliary global variables
  init_block(program, fout);
  VerifyModule(program);

  // insert sync before each vote, and replace the
  // original vote function to warp vote
  DEBUG_INFO("CuPBoP pass: handle warp vote\n");
  handle_warp_vote(program);
  VerifyModule(program);

  // replace warp shuffle
  DEBUG_INFO("CuPBoP pass: handle warp shuffle\n");
  handle_warp_shfl(program);
  VerifyModule(program);

  // insert sync
  DEBUG_INFO("CuPBoP pass: insert barrier\n");
  insert_sync(program);
  VerifyModule(program);

  // split block by sync
  DEBUG_INFO("CuPBoP pass: split block by sync\n");
  split_block_by_sync(program);
  VerifyModule(program);

  // add loop for intra&intera thread, it refers 'hierarchical collapsing' in
  // COX paper.
  DEBUG_INFO("CuPBoP pass: insert warp loop\n");
  insert_warp_loop(program);
  VerifyModule(program);

  DEBUG_INFO("CuPBoP pass: replace built-in function\n");
  replace_built_in_function(program);
  VerifyModule(program);

  // the input kernel programs have NVIDIA metadata, they need to be replaced to
  // CPU metadata
  DEBUG_INFO("CuPBoP pass: generate cpu format\n");
  generate_cpu_format(program);
  VerifyModule(program);

  DEBUG_INFO("CuPBoP pass: performance optimization\n");
  performance_optimization(program);

  VerifyModule(program);

  DumpModule(program, argv[2]);

  fout.close();
  return 0;
}
