#include "performance.h"
#include "replace_block_size.h"
#include "tool.h"
#include "llvm/IR/Module.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <set>
#include <stdlib.h>

using namespace llvm;

// Replace Block_size with given constant, apply O3 optimizations, and generate
// shared library
int main(int argc, char **argv) {
  // args: Module_path, output_path, BLOCK_X, BLOCK_Y, BLOCK_Z
  assert(argc == 6 && "incorrect number of arguments\n");
  llvm::Module *program = LoadModuleFromFilr(argv[1]);
  char *output_path = argv[2];
  int block_x = atoi(argv[3]);
  int block_y = atoi(argv[4]);
  int block_z = atoi(argv[5]);

  // replace the block_size with constant
  replace_block_size_global_variable(program, block_x, block_y, block_z);

  // apply O3 optimizations and other optimizations
  performance_optimization(program);
  VerifyModule(program);

  // generate shared library
  program->dump();

  return 0;
}
