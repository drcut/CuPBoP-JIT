#include "exec.hpp"
#include "performance.h"
#include "replace_configuration.h"
#include "tool.h"
#include "llvm/IR/Module.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string>

using namespace llvm;

// Replace Block_size with given constant, apply O3 optimizations, and generate
// shared library
int main(int argc, char **argv) {
  // args: Module_path, GRID_X, GRID_Y, GRID_Z, BLOCK_X, BLOCK_Y, BLOCK_Z
  assert(argc == 8 && "incorrect number of arguments\n");
  llvm::Module *program = LoadModuleFromFilr(argv[1]);
  int grid_x = atoi(argv[2]);
  int grid_y = atoi(argv[3]);
  int grid_z = atoi(argv[4]);
  int block_x = atoi(argv[5]);
  int block_y = atoi(argv[6]);
  int block_z = atoi(argv[7]);

  // replace the block_size with constant
  replace_grid_block_size_global_variable(program, grid_x, grid_y, grid_z,
                                          block_x, block_y, block_z);
  VerifyModule(program);

  // apply O3 optimizations and other optimizations
  performance_optimization(program);
  VerifyModule(program);

  // generate bitcode file
  DumpModule(program, "/tmp/cache/tmp.bc");
  // generate object file (.o)
  // object file name: kernel1_kernel2_..._block_x_bloxk_y_block_z.o
  std::string object_file_name = "/tmp/cache/";
  for (Module::iterator i = program->begin(), e = program->end(); i != e; ++i) {
    Function *F = &(*i);
    if (isKernelFunction(program, F))
      object_file_name += '_' + F->getName().str();
  }
  object_file_name += '_' + std::to_string(block_x) + '_' +
                      std::to_string(block_y) + '_' + std::to_string(block_z);
  object_file_name += ".so";
  printf("object_file_name: %s\n", object_file_name.c_str());
  // use clang to generate shared library
  std::stringstream ss;
  ss << "clang -shared "
     << "/tmp/cache/tmp.bc"
     << " -o " << object_file_name;
  exec(ss.str().c_str());
  return 0;
}
