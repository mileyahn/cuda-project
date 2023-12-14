#include <math.h>
#include <mpi.h>
#include <cassert>
#include <thread>

#include "classifier.h"
#include "util.h"

static int mpi_rank;

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define DEBUG 0
#define BATCH 2
#define TSIZE 2
#define RSIZE 16
#define NGPU 4
#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  int num_elem();
  void fill_zeros();

  float *buf = nullptr;
  float *gbuf[NGPU] = {nullptr, nullptr, nullptr, nullptr};
  int ndim = 0;
  int shape[4];

  void toCPU();
  void toGPU();
};

Tensor::Tensor(std::vector<int> shape_) {
  // reshape
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  //reshape fin

  CHECK_CUDA(cudaMallocHost(&buf, N_ * sizeof(float)));
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&gbuf[i], N_ * sizeof(float)));
  }
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  // reshape
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  // reshape fin

  CHECK_CUDA(cudaMallocHost(&buf, N_ * sizeof(float)));

  memcpy(buf, buf_, N_ * sizeof(float));
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&gbuf[i], N_ * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(gbuf[i], buf, N_ * sizeof(float), cudaMemcpyHostToDevice));
  }
}

void Tensor::toCPU(){
  CHECK_CUDA(cudaMemcpy(buf, gbuf, num_elem() * sizeof(float), cudaMemcpyDeviceToHost));
}

void Tensor::toGPU(){
  CHECK_CUDA(cudaMemcpy(gbuf, buf, num_elem() * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::~Tensor() {
  if (buf != nullptr) CHECK_CUDA(cudaFreeHost(buf));
  for (size_t i = 0; i < NGPU; i++) {
    if (gbuf[i] != nullptr) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaFree(gbuf[i]));
    }
  }
}

int Tensor::num_elem() {
  int sz = 1;
  for (int i = 0; i < ndim; ++i) { sz *= shape[i]; }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n = 0; n < N_; ++n) { buf[n] = 0.0; }
}

// Parameters
Tensor *w_conv1, *w_conv2, *w_conv3, *w_conv4, *w_conv5, *w_conv6, *b_conv1,
    *b_conv2, *b_conv3, *b_conv4, *b_conv5, *b_conv6, *w_fc1, *w_fc2, *w_fc3,
    *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

// Activations
Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
Tensor *a_conv2, *a_relu2, *a_pool2;
Tensor *a_conv3, *a_relu3;
Tensor *a_conv4, *a_relu4;
Tensor *a_conv5, *a_relu5;
Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
Tensor *a_collapse;
Tensor *a_linear1, *a_relu7;
Tensor *a_linear2, *a_relu8;
Tensor *a_linear3;
//me
Tensor *a_output;
Tensor *a_input;

// Operations
void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, int gpuIdx,
            int stride, int padding, int dilation, bool has_bias);
void relu(Tensor *input, Tensor *output, int gpuIdx);
void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride, int gpuIdx);
void collapse(Tensor *input, Tensor *output, int gpuIdx);
void matmul(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias, int gpuIdx);
void vector_sum(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias, int gpuIdx);
void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output, int gpuIdx);
//me
void find_maxIdx(Tensor *input, Tensor *output, int idx, int N, int gpuIdx);

void check(Tensor *t){
  t->toCPU();
  for(int i = 0; i< t->num_elem(); i++){
    printf(" %f", t->buf[i]);
  }
}

// Only the first process (root, mpi_rank == 0) has the input and output
// Parallelization method is totally up to you, but you should gather 
// the output at rank 0
void model_thread(float *input_, float *output_, int gpuIdx, int N) {
  if (mpi_rank == 0) {
    CHECK_CUDA(cudaSetDevice(gpuIdx));
    for (int idx = 0; idx < N / NGPU; idx += BATCH) {

      CHECK_CUDA(cudaMemcpy(a_input->gbuf[gpuIdx], input_ + idx * VOCAB_SIZE * MAX_LENGTH, BATCH * VOCAB_SIZE * MAX_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
      
      // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
      conv1d(a_input, w_conv1, b_conv1, a_conv1, gpuIdx, 1, 0, 1, true);
      
      layernorm(a_conv1, gamma_conv1, beta_conv1, a_layernorm1, gpuIdx);
      relu(a_layernorm1, a_relu1, gpuIdx);
      maxpool1d(a_relu1, a_pool1, 3, 3, gpuIdx);

      // Conv block 2 : Conv1d + ReLU + MaxPool1d
      conv1d(a_pool1, w_conv2, b_conv2, a_conv2, gpuIdx, 1, 0, 1, true);
      relu(a_conv2, a_relu2, gpuIdx);
      maxpool1d(a_relu2, a_pool2, 3, 3, gpuIdx);
      
      // Conv block 3 : Conv1d + ReLU
      conv1d(a_pool2, w_conv3, b_conv3, a_conv3, gpuIdx, 1, 0, 1, true);
      relu(a_conv3, a_relu3, gpuIdx);
      
      // Conv block 4 : Conv1d + ReLU
      conv1d(a_relu3, w_conv4, b_conv4, a_conv4, gpuIdx, 1, 0, 1, true);
      relu(a_conv4, a_relu4, gpuIdx);

      // Conv block 5 : Conv1d + ReLU
      conv1d(a_relu4, w_conv5, b_conv5, a_conv5, gpuIdx, 1, 0, 1, true);
      relu(a_conv5, a_relu5, gpuIdx);

      // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
      conv1d(a_relu5, w_conv6, b_conv6, a_conv6, gpuIdx, 1, 0, 1, true);
      layernorm(a_conv6, gamma_conv6, beta_conv6, a_layernorm6, gpuIdx);
      relu(a_layernorm6, a_relu6, gpuIdx);
      maxpool1d(a_relu6, a_pool6, 3, 3, gpuIdx);
      
      // Collapse
      collapse(a_pool6, a_collapse, gpuIdx);
      // FC block 1 : Linear + ReLU
      matmul(a_collapse, w_fc1, b_fc1, a_linear1, true, gpuIdx);
      relu(a_linear1, a_relu7, gpuIdx);

      // FC block 2 : Linear + ReLU
      matmul(a_relu7, w_fc2, b_fc2, a_linear2, true, gpuIdx);
      relu(a_linear2, a_relu8, gpuIdx);

      // FC block 3 : Linear
      vector_sum(a_relu8, w_fc3, b_fc3, a_linear3, true, gpuIdx);

      find_maxIdx(a_linear3, a_output, idx, N, gpuIdx);

      CHECK_CUDA(cudaMemcpy(output_ + idx, a_output->gbuf[gpuIdx], BATCH * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }    // if mpi_rank == 0
}

void classifier(float *input_, float *output_, int N) {
  std::thread threads[NGPU];
  for (int gpuIdx = 0; gpuIdx < NGPU; gpuIdx++) {
    int inPos = (N / NGPU) * gpuIdx; // N is always multiple of 4
    threads[gpuIdx] = std::thread(model_thread, input_ + inPos * VOCAB_SIZE * MAX_LENGTH,
                                  output_ + inPos, gpuIdx, N);
  }

  for (int gpuIdx = 0; gpuIdx < NGPU; gpuIdx++) {
    threads[gpuIdx].join();
  }
}

__global__ void conv1d_kernel(float *in, float *out, float *weight, float *bias, int out_channels, int in_channels, int kernel_size, int input_length, int output_length, bool has_bias){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = tidx / (out_channels * output_length);
  int oc = (tidx / output_length) % out_channels;
  int ol = tidx % output_length;

  if(oc >= out_channels || ol >= output_length) return;  
  
  float val = 0.0f;
  int offset = ol;
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int ks = 0; ks < kernel_size; ++ks) {
      val += weight[oc * in_channels * kernel_size + ic * kernel_size + ks] *
                 in[b * in_channels * input_length + ic * input_length + ks + offset];
    }
  }
  if (has_bias) val += bias[oc];
  out[b * out_channels * output_length + oc * output_length + ol] = val;  
}

void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, int gpuIdx,
            int stride = 1, int padding = 0, int dilation = 1,
            bool has_bias = true) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  float *w = weight->gbuf[gpuIdx];
  float *b = bias->gbuf[gpuIdx];

  int out_channels = weight->shape[0];
  int in_channels = weight->shape[1];
  int kernel_size = weight->shape[2];
  int input_length = input->shape[3];
  int output_length =
      (input->shape[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  int total_threads = BATCH * out_channels * output_length;
  int block_size = 1024; 
  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);
  conv1d_kernel<<<gridDim, blockDim>>>(in, out, w, b, out_channels, in_channels, kernel_size, input_length, output_length, has_bias);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void relu_kernel(float *in, float *out, int N){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tidx >= N) return;
  out[tidx] = fmaxf(in[tidx], 0.0f);
}

void relu(Tensor *input, Tensor *output, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  int N = input->num_elem();

  relu_kernel<<<CEIL_DIV(N, 256), 256>>>(in, out, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void maxpool1d_kernel(float *in, float *out, int IL, int OC, int OL, int kernel_size, int stride){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = tidx / (OL * OC);
  int oc = (tidx / OL) % OC;
  int ol = tidx % OL;
  if(oc >= OC || ol >= OL) return;

  int i_idx = b * OC * IL + oc * IL + ol * stride;
  int o_idx = b * OC * OL + oc * OL + ol;
  float mx = -1e99;
  for (int ks = 0; ks < kernel_size; ++ks) {
    float val = in[i_idx + ks];
    if (val > mx) mx = val;
  }
  out[o_idx] = mx;
}

void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];

  int IL = input->shape[3];
  int OC = output->shape[2];
  int OL = output->shape[3];

  int total_threads = BATCH * OC * OL;
  int block_size = 512;
  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);
  maxpool1d_kernel<<<gridDim, blockDim>>>(in, out, IL, OC, OL, kernel_size, stride);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void collapse_kernel(float *in, float *out, int N){
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if(n >= N) return;
  out[n] = in[n];
}

void collapse(Tensor *input, Tensor *output, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  int N = input->num_elem();

  collapse_kernel<<<CEIL_DIV(N, 256), 256>>>(in, out, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void matmul_kernel(float *A, float *B, float *C, float *bias, int K, int M, int N, bool has_bias){
  int gidC = blockDim.x * blockIdx.x + threadIdx.x;
  int gidR = blockDim.y * blockIdx.y + threadIdx.y;
  int tidC = threadIdx.x;
  int tidR = threadIdx.y;

  __shared__ float Asub[TSIZE][TSIZE];
  __shared__ float Bsub[TSIZE][TSIZE];

  float sum = 0.0f;
  if (has_bias) sum += bias[gidC];

  for (int offset = 0; offset < K; offset += TSIZE) {
    Asub[tidR][tidC] = (offset + tidC < K) ? A[gidR * K + offset + tidC] : 0;
    Bsub[tidR][tidC] = (offset + tidR < K) ? B[gidC * K + offset + tidR] : 0;

    __syncthreads();

    for (int k = 0; k < TSIZE; k++) {
      sum += Asub[tidR][k] * Bsub[k][tidC];
    }

    __syncthreads();
  }
  C[gidR * M + gidC] = sum;// > 0 ? sum : 0;
}

void matmul(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  float *w = weight->gbuf[gpuIdx];
  float *b = bias->gbuf[gpuIdx];

  int K = input->shape[2];
  int M = output->shape[2];
  int N = BATCH;

  dim3 block(TSIZE, TSIZE);
  dim3 grid(CEIL_DIV(M, TSIZE), CEIL_DIV(N, TSIZE));
  matmul_kernel<<<grid, block>>>(in, w, out, b, K, M, N, has_bias);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void vector_sum_kernel(float *in, float *out, float *weight, float *bias, int K, int M, bool has_bias){
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int tidx = threadIdx.x;

  if(k >= K) return;

  __shared__ float o1[RSIZE];
  __shared__ float o2[RSIZE];
  __shared__ float o3[RSIZE];
  __shared__ float o4[RSIZE];

  float input = in[b * K + k];
  o1[tidx] = input * weight[k];
  o2[tidx] = input * weight[K + k];
  o3[tidx] = input * weight[2 * K + k];
  o4[tidx] = input * weight[3 * K + k];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tidx < stride) {
      o1[tidx] += o1[tidx + stride];
      o2[tidx] += o2[tidx + stride];
      o3[tidx] += o3[tidx + stride];
      o4[tidx] += o4[tidx + stride];
    }
    __syncthreads();
  }

  if (tidx == 0) {
    atomicAdd(&out[b * M], o1[0]);
    atomicAdd(&out[b * M + 1], o2[0]);
    atomicAdd(&out[b * M + 2], o3[0]);
    atomicAdd(&out[b * M + 3], o4[0]);
    if (blockIdx.x == 0 && has_bias) {
      atomicAdd(&out[b * M], bias[0]);
      atomicAdd(&out[b * M + 1], bias[1]);
      atomicAdd(&out[b * M + 2], bias[2]);
      atomicAdd(&out[b * M + 3], bias[3]);
    }
  }
}

void vector_sum(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  float *w = weight->gbuf[gpuIdx];
  float *b = bias->gbuf[gpuIdx];

  int K = input->shape[2]; //1024
  int M = output->shape[2]; // 4
  int B = BATCH;

  CHECK_CUDA(cudaMemset(out, 0, output->num_elem() * sizeof(float)));
  
  dim3 block(RSIZE, 1);
  dim3 grid(CEIL_DIV(K, RSIZE), B);
  vector_sum_kernel<<<grid, block>>>(in, out, w, b, K, M, has_bias);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void layernorm_kernel(float *in, float *out, float *gamma, float *bias, int N){
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = b * N;

  // E[X], E[X^2]
  float sum1 = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < N; ++i) {
      sum1 += in[idx + i];
      sum2 += in[idx + i] * in[idx + i];
  }
  float mean1 = sum1 / (float)N;
  float mean2 = sum2 / (float)N;

  // V[X]
  float var = mean2 - mean1 * mean1;  

  // Normalization
  for (int i = 0; i < N; ++i) {
    out[idx + i] = (in[idx + i] - mean1) / sqrtf(var + 1e-5) * gamma[i] + bias[i];
  }
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  float *g = gamma->gbuf[gpuIdx];
  float *b = beta->gbuf[gpuIdx];
  int N = input->num_elem() / BATCH;

  dim3 block(1, 1);
  dim3 grid(1, BATCH);
  layernorm_kernel<<<grid, block>>>(in, out, g, b, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void find_maxIdx_kernel(float *in, float *out, int N, int idx, int num){
  int b = blockDim.y * blockIdx.y + threadIdx.y;

  // if(idx * BATCH + b > N) return;

  float max_val = -1e99f;
  int max_idx = 0;
  for (int i = 0; i < num; ++i) {
    if (in[b * num + i] > max_val) {
      max_val = in[b * num + i];
      max_idx = i;
    }
  }
  out[b] = max_idx;
}

void find_maxIdx(Tensor *input, Tensor *output, int idx, int N, int gpuIdx) {
  float *in = input->gbuf[gpuIdx];
  float *out = output->gbuf[gpuIdx];
  int num = input->num_elem() / BATCH;

  dim3 block(1, 1);
  dim3 grid(1, BATCH);
  find_maxIdx_kernel<<<grid, block>>>(in, out, N, idx, num);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

// load the parameter binary file and store parameters into Tensors
// Only the first process (root, mpi_rank == 0) has the parameter
// You must broadcast it to the others
void initialize_classifier(float *parameter, int N) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    w_conv1 = new Tensor({256, 70, 7}, parameter + OFFSET0);
    b_conv1 = new Tensor({256}, parameter + OFFSET1);
    gamma_conv1 = new Tensor({256, 1008}, parameter + OFFSET2);
    beta_conv1 = new Tensor({256, 1008}, parameter + OFFSET3);
    w_conv2 = new Tensor({256, 256, 7}, parameter + OFFSET4);
    b_conv2 = new Tensor({256}, parameter + OFFSET5);
    w_conv3 = new Tensor({256, 256, 3}, parameter + OFFSET6);
    b_conv3 = new Tensor({256}, parameter + OFFSET7);
    w_conv4 = new Tensor({256, 256, 3}, parameter + OFFSET8);
    b_conv4 = new Tensor({256}, parameter + OFFSET9);
    w_conv5 = new Tensor({256, 256, 3}, parameter + OFFSET10);
    b_conv5 = new Tensor({256}, parameter + OFFSET11);
    w_conv6 = new Tensor({256, 256, 3}, parameter + OFFSET12);
    b_conv6 = new Tensor({256}, parameter + OFFSET13);
    gamma_conv6 = new Tensor({256, 102}, parameter + OFFSET14);
    beta_conv6 = new Tensor({256, 102}, parameter + OFFSET15);
    w_fc1 = new Tensor({1024, 8704}, parameter + OFFSET16);
    b_fc1 = new Tensor({1024}, parameter + OFFSET17);
    w_fc2 = new Tensor({1024, 1024}, parameter + OFFSET18);
    b_fc2 = new Tensor({1024}, parameter + OFFSET19);
    w_fc3 = new Tensor({4, 1024}, parameter + OFFSET20);
    b_fc3 = new Tensor({4}, parameter + OFFSET21);

    a_conv1 = new Tensor({BATCH, 1, 256, 1008});
    a_layernorm1 = new Tensor({BATCH, 1, 256, 1008});
    a_relu1 = new Tensor({BATCH, 1, 256, 1008});
    a_pool1 = new Tensor({BATCH, 1, 256, 336});
    a_conv2 = new Tensor({BATCH, 1, 256, 330});
    a_relu2 = new Tensor({BATCH, 1, 256, 330});
    a_pool2 = new Tensor({BATCH, 1, 256, 110});
    a_conv3 = new Tensor({BATCH, 1, 256, 108});
    a_relu3 = new Tensor({BATCH, 1, 256, 108});
    a_conv4 = new Tensor({BATCH, 1, 256, 106});
    a_relu4 = new Tensor({BATCH, 1, 256, 106});
    a_conv5 = new Tensor({BATCH, 1, 256, 104});
    a_relu5 = new Tensor({BATCH, 1, 256, 104});
    a_conv6 = new Tensor({BATCH, 1, 256, 102});
    a_layernorm6 = new Tensor({BATCH, 1, 256, 102});
    a_relu6 = new Tensor({BATCH, 1, 256, 102});
    a_pool6 = new Tensor({BATCH, 1, 256, 34});
    a_collapse = new Tensor({BATCH, 1, 8704});
    a_linear1 = new Tensor({BATCH, 1, 1024});
    a_relu7 = new Tensor({BATCH, 1, 1024});
    a_linear2 = new Tensor({BATCH, 1, 1024});
    a_relu8 = new Tensor({BATCH, 1, 1024});
    a_linear3 = new Tensor({BATCH, 1, 4});

    //yelim
    a_input = new Tensor({BATCH, 1, VOCAB_SIZE, MAX_LENGTH});
    a_output = new Tensor({BATCH, 1});
  }
}

// Free all dynamically allocated variables
void finalize_classifier() {
  if (mpi_rank == 0) {
    delete w_conv1;
    delete b_conv1;
    delete w_conv2;
    delete b_conv2;
    delete w_conv3;
    delete b_conv3;
    delete w_conv4;
    delete b_conv4;
    delete w_conv5;
    delete b_conv5;
    delete w_conv6;
    delete b_conv6;
    delete w_fc1;
    delete b_fc1;
    delete w_fc2;
    delete b_fc2;
    delete w_fc3;
    delete b_fc3;
    delete gamma_conv1;
    delete gamma_conv6;
    delete beta_conv1;
    delete beta_conv6;
    delete a_conv1;
    delete a_layernorm1;
    delete a_relu1;
    delete a_pool1;
    delete a_conv2;
    delete a_relu2;
    delete a_pool2;
    delete a_conv3;
    delete a_relu3;
    delete a_conv4;
    delete a_relu4;
    delete a_conv5;
    delete a_relu5;
    delete a_conv6;
    delete a_layernorm6;
    delete a_relu6;
    delete a_pool6;
    delete a_collapse;
    delete a_linear1;
    delete a_relu7;
    delete a_linear2;
    delete a_relu8;
    delete a_linear3;
    //yelim
    delete a_output;
  }
}