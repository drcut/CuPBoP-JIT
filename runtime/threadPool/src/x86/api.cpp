#include "api.h"
#include "blockingconcurrentqueue.h"
#include "debug.hpp"
#include "def.h"
#include "macros.h"
#include "structures.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <omp.h>

// Create Kernel
static int kernelIds = 0;
cu_kernel *create_kernel(const void *func, dim3 gridDim, dim3 blockDim,
                         void **args, size_t sharedMem, cudaStream_t stream) {
  cu_kernel *ker = (cu_kernel *)calloc(1, sizeof(cu_kernel));

  // set the function pointer
  ker->start_routine = (void *(*)(void *))func;
  ker->args = args;

  ker->gridDim = gridDim;
  ker->blockDim = blockDim;
  ker->shared_mem = sharedMem;
  ker->stream = stream;
  ker->totalBlocks = gridDim.x * gridDim.y * gridDim.z;
  ker->blockSize = blockDim.x * blockDim.y * blockDim.z;
  return ker;
}

// scheduler
static cu_pool *scheduler;

__thread int block_size = 0;
__thread int block_size_x = 0;
__thread int block_size_y = 0;
__thread int block_size_z = 0;
__thread int grid_size_x = 0;
__thread int grid_size_y = 0;
__thread int grid_size_z = 0;
__thread int block_index = 0;
__thread int block_index_x = 0;
__thread int block_index_y = 0;
__thread int block_index_z = 0;
__thread int thread_memory_size = 0;
__thread int *dynamic_shared_memory = NULL;
__thread int warp_shfl[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
int** global_dynamic_shared_memory;


/*
  Kernel Launch with numBlocks and numThreadsPerBlock
*/
int cuLaunchKernel(cu_kernel **k) {
  // Calculate Block Size N/numBlocks
  cu_kernel *ker = *k;

  int dynamic_shared_mem_size = ker->shared_mem;
  dim3 gridDim= ker->gridDim;
  dim3 blockDim= ker->blockDim;
  
  #pragma omp parallel for schedule(static)
  for(int block_index = 0; block_index < ker->totalBlocks; block_index++)
  {
      block_size = ker->blockSize;
      block_size_x = blockDim.x;
      block_size_y = blockDim.y;
      block_size_z = blockDim.z;
      grid_size_x = gridDim.x;
      grid_size_y = gridDim.y;
      grid_size_z = gridDim.z;

      if (dynamic_shared_mem_size > 0) {
        dynamic_shared_memory = (int *)malloc(dynamic_shared_mem_size);
      }
      int tmp = block_index;
      block_index_x = tmp / (grid_size_y * grid_size_z);
      tmp = tmp % (grid_size_y * grid_size_z);
      block_index_y = tmp / (grid_size_z);
      tmp = tmp % (grid_size_z);
      block_index_z = tmp;
      ker->start_routine(ker->args);
  }
  return 0;
}

/*
    Thread Gets Work
*/
int get_work(c_thread *th) {
  int dynamic_shared_mem_size = 0;
  dim3 gridDim;
  dim3 blockDim;
  while (true) {
    // try to get a task from the queue
    cu_kernel *k;
    th->busy = scheduler->kernelQueue->wait_dequeue_timed(
        k, std::chrono::milliseconds(5));
    if (th->busy) {
      // set runtime configuration
      gridDim = k->gridDim;
      blockDim = k->blockDim;
      dynamic_shared_mem_size = k->shared_mem;
      block_size = k->blockSize;
      block_size_x = blockDim.x;
      block_size_y = blockDim.y;
      block_size_z = blockDim.z;
      grid_size_x = gridDim.x;
      grid_size_y = gridDim.y;
      grid_size_z = gridDim.z;
      if (dynamic_shared_mem_size > 0)
        dynamic_shared_memory = (int *)malloc(dynamic_shared_mem_size);
      // execute GPU blocks
      for (block_index = k->startBlockId; block_index < k->endBlockId + 1;
           block_index++) {
        int tmp = block_index;
        block_index_x = tmp / (grid_size_y * grid_size_z);
        tmp = tmp % (grid_size_y * grid_size_z);
        block_index_y = tmp / (grid_size_z);
        tmp = tmp % (grid_size_z);
        block_index_z = tmp;
        k->start_routine(k->args);
      }
      th->completeTask++;
    }
    // if cannot get tasks, check whether programs stop
    if (scheduler->threadpool_shutdown_requested) {
      return true; // thread exit
    }
  }
  return 0;
}

void *driver_thread(void *p) {
  struct c_thread *td = (struct c_thread *)p;
  int is_exit = 0;
  td->exit = false;
  td->busy = false;
  // get work
  is_exit = get_work(td);

  // exit the routine
  if (is_exit) {
    td->exit = true;
    pthread_exit(NULL);
  } else {
    assert(0 && "driver thread stop incorrectly\n");
  }
}

void scheduler_uninit() { assert(0 && "Scheduler Unitit no Implemente\n"); }

/*
  Barrier for Kernel Launch
*/
void cuSynchronizeBarrier() {
  return;
}
