#ifndef __NVVM2x86_ANTI_COALESCING_OPTIMIZATION__
#define __NVVM2x86_ANTI_COALESCING_OPTIMIZATION__

#include "llvm/IR/Function.h"
/*
 *The CUDA global memory coalescing optimization will result to low cache hit
 *rate on CPU. Thus, we need to implement transformation.
 * For example:
 * Input CUDA:
 *   uint32_t index = tid;
 *   while (index < num_pixels) {
 *      uint32_t color = pixels[index];
 *      priv_hist[color]++;
 *      index += gsize;
 * }
 * Output CUDA:
 *   uint32_t index = tid;
 *   __shared__ bool has_activated_thread;
 *   bool thread_activated = true;
 *   do {
 *      has_activated_thread = false;
 *      __syncthreads();
 *      thread_activated = thread_activated & (index < num_pixels);
 *      has_activated_thread |= thread_activated;
 *      if (thread_activated) {
 *        uint32_t color = pixels[index];
 *        priv_hist[color]++;
 *        index += gsize;
 *      }
 *   } while (has_activated_thread);
 *   __syncthreads();
 */
void anti_global_mem_coalescing_optimization(llvm::Module *M);

#endif
