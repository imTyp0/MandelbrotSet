#include<cuda_runtime.h>
#include<iostream>
#include<fstream>
#include<cmath>
#include<chrono>
#include<thread>

const int MAX_ITERATIONS = 100000;
// Ratio of 4:3
const int WIDTH = 2560*16;
const int HEIGHT = 1920*16;

// Maps a number from a range to another
__device__
double map(double n, double oldLow, double oldHigh, double newLow, double newHigh){
  return newLow + (n - oldLow) * (newHigh - newLow) / (oldHigh - oldLow);
}

__device__
unsigned char isMandelbrot(double x, double y){
  double zx = 0.0;
  double zy = 0.0;
  int iteration = 0;
  
  // While we're still in the mandelbrot set: ||z|| <= 4
  // z_(n+1) = (z_n)^2 + c, sub in z_n = zx + izy and c = x + iy 
  while ((zx*zx + zy*zy) <= 4.0 && iteration < MAX_ITERATIONS){
    double xtemp = zx * zx - zy * zy + x; // New real part of z
    zy = 2.0 * zx * zy + y; // New imaginary part of z
    zx = xtemp;
    iteration++;
  }

  // Speed of divergence = iteration/MAX_ITERATIONS \in [0, 1]
  // 0: diverged instantly, 1: converged
  // 1 - SoD = divergenceFactor \in [1, 0]
  // Color: [0-255], 0: black, 255: white
  
  double divergenceFactor = 1.0 - (double) iteration / (double) MAX_ITERATIONS;
  divergenceFactor = map(std::pow(divergenceFactor, 256), 0, 1, 0, 255);
  if (iteration == MAX_ITERATIONS) divergenceFactor = 0;

  return divergenceFactor;
}

__global__
void mandelbrotKernel(
  double* rX, double* rY, unsigned char* out
){
  // Get thread id stuff and for each of those, run over rangeX and rangeY
  // to compute isMandelbrot on the x,y pair
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < WIDTH && y < HEIGHT){
    out[y * WIDTH + x] = isMandelbrot(rX[x], rY[y]);
  }
}

int main(){

  
  // Image body
  double* h_rangeX = new double[WIDTH];
  double* h_rangeY = new double[HEIGHT];
  unsigned char* h_image = new unsigned char[WIDTH * HEIGHT];

  // Fill the ranges
  // X: [-2.25, 0.75]
  for (int i = 0; i < WIDTH; i++){
    h_rangeX[i] = -2.25 + 3.0 * i / (WIDTH-1);
  }
  // Y: [-1.25, 1.25]
  for (int i = 0; i < HEIGHT; i++){
    h_rangeY[i] = -1.25 + 2.5 * i / (HEIGHT-1);
  }
  
  // Allocate memory
  double* d_rangeX;
  double* d_rangeY;
  unsigned char* d_image;

  cudaMalloc(&d_rangeX, WIDTH * sizeof(double));
  cudaMalloc(&d_rangeY, HEIGHT * sizeof(double));
  cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
  
  cudaMemcpy(d_rangeX, h_rangeX, WIDTH * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_rangeY, h_rangeY, HEIGHT * sizeof(double),
             cudaMemcpyHostToDevice);

  // Run the function on the GPU
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  // Create a thread to pass device data back to the host asynchronously
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto before = std::chrono::high_resolution_clock::now();

  mandelbrotKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
    d_rangeX, d_rangeY, d_image
  );
  
  cudaMemcpyAsync(h_image, d_image, WIDTH*HEIGHT*sizeof(unsigned char),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  // Write to file
  std::ofstream filp("image.ppm");
  if (!filp){
    std::cerr << "Error opening the file" << std::endl;
    return 1;
  }

  // Image header
  filp << "P5\n" << WIDTH << " " << HEIGHT << "\n255" << std::endl; 
  // Image contents
  filp.write(reinterpret_cast<const char*>(h_image), WIDTH*HEIGHT*sizeof(unsigned char));

  auto after = std::chrono::high_resolution_clock::now();
  auto millis =
    std::chrono::duration_cast<std::chrono::milliseconds>(after-before).count();

  printf("\rDone. Took %im, %is, %ims", millis/60000, (millis%60000)/1000, millis%1000);
  
  // Free the memory
  cudaFree(d_rangeX);
  cudaFree(d_rangeY);
  cudaFree(d_image);
  cudaStreamDestroy(stream);
  delete[] h_rangeX;
  delete[] h_rangeY;
  delete[] h_image;

  filp.close();
}
