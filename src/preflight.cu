#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

__global__ void fill(uint32_t *p, size_t n, uint32_t seed) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x)
    p[i] = uint32_t(i * 2654435761u) ^ 0xa5a55a5au ^ seed;
}
static bool valid(const std::vector<uint32_t>& x, uint32_t seed) {
  for (size_t i=0; i<x.size(); ++i) if(x[i] != (uint32_t(i * 2654435761u) ^ 0xa5a55a5au ^ seed)) return false;
  return true;
}
int main() {
  int n=0, driver=0, runtime=0;
  CUDA(cudaGetDeviceCount(&n)); CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  if(n<1) return 2;
  constexpr size_t bytes=128*1024*1024, count=bytes/sizeof(uint32_t);
  constexpr int warmup=5, iters=20;
  std::vector<uint32_t*> src(n), dst(n);
  std::vector<uint32_t> host(count);
  printf("{\"device_count\":%d,\"driver\":%d,\"runtime\":%d,\"bytes\":%zu,\"warmup\":%d,\"iterations\":%d,\"devices\":[",n,driver,runtime,bytes,warmup,iters);
  for(int d=0;d<n;d++) {
    CUDA(cudaSetDevice(d));
    cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop,d));
    CUDA(cudaMalloc(&src[d],bytes)); CUDA(cudaMalloc(&dst[d],bytes));
    fill<<<256,256>>>(src[d],count,uint32_t(d+1)*2246822519u); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    for(int k=0;k<warmup;k++) CUDA(cudaMemcpyAsync(dst[d],src[d],bytes,cudaMemcpyDeviceToDevice));
    CUDA(cudaEventRecord(start));
    for(int k=0;k<iters;k++) CUDA(cudaMemcpyAsync(dst[d],src[d],bytes,cudaMemcpyDeviceToDevice));
    CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
    float ms; CUDA(cudaEventElapsedTime(&ms,start,stop));
    CUDA(cudaMemcpy(host.data(),dst[d],bytes,cudaMemcpyDeviceToHost));
    if(!valid(host,uint32_t(d+1)*2246822519u)) { fprintf(stderr,"device memory mismatch\n"); return 1; }
    printf("%s{\"id\":%d,\"name\":\"%s\",\"sm_count\":%d,\"memory_bytes\":%zu,\"d2d_payload_GBs\":%.6f}",d?",":"",d,prop.name,prop.multiProcessorCount,prop.totalGlobalMem,bytes*iters/(ms*1e6));
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
  }
  printf("],\"peer_links\":[");
  bool first=true;
  for(int s=0;s<n;s++) for(int d=0;d<n;d++) if(s!=d) {
    int capable=0; CUDA(cudaDeviceCanAccessPeer(&capable,s,d));
    printf("%s{\"src\":%d,\"dst\":%d,\"p2p_supported\":%s",first?"":",",s,d,capable?"true":"false");first=false;
    CUDA(cudaSetDevice(s));
    if(capable) { CUDA(cudaDeviceEnablePeerAccess(d,0)); }
    cudaEvent_t start, stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    for(int k=0;k<warmup;k++) CUDA(cudaMemcpyPeerAsync(dst[d],d,src[s],s,bytes));
    CUDA(cudaEventRecord(start));
    for(int k=0;k<iters;k++) CUDA(cudaMemcpyPeerAsync(dst[d],d,src[s],s,bytes));
    CUDA(cudaEventRecord(stop));CUDA(cudaEventSynchronize(stop));
    float ms;CUDA(cudaEventElapsedTime(&ms,start,stop));
    CUDA(cudaSetDevice(d));CUDA(cudaMemcpy(host.data(),dst[d],bytes,cudaMemcpyDeviceToHost));
    if(!valid(host,uint32_t(s+1)*2246822519u)) return 1;
    printf(",\"correctness\":\"pass\",\"payload_GBs\":%.6f}",bytes*iters/(ms*1e6));
    CUDA(cudaSetDevice(s));CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(stop));
  }
  for(int d=0;d<n;d++){CUDA(cudaSetDevice(d));CUDA(cudaFree(src[d]));CUDA(cudaFree(dst[d]));}
  printf("],\"correctness\":\"pass\"}\n");return 0;
}
