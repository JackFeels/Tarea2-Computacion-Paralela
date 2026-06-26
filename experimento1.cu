// =============================================================================
//  Tarea 2 - Introduccion a la Computacion Paralela
//  Experimento 1: implementacion tradicional en CUDA.
//
//  En esta version asumo que todo el dataset entra contiguo en la memoria de la
//  GPU, asi que cargo todo de una con el stream por defecto. Para calcular la
//  matriz de covarianza C (n x n) sigo tres pasos: primero el vector promedio,
//  despues centro los datos restandole ese promedio, y por ultimo hago la
//  multiplicacion matricial Vbar^T * Vbar con tiling en memoria compartida.
//  Mido por separado la copia H2D, el computo de los kernels y la copia D2H de C
//  para ver donde se va realmente el tiempo.
//
//  Compilar:  nvcc -O3 -o exp1 experimento1.cu -lpng -ljpeg -lz -lpthread -I.
//  Ejecutar:  ./exp1 <directorio_imagenes> [maxImagenes]
// =============================================================================

#define cimg_display 0       // no uso la parte grafica de CImg
#define cimg_use_png         // para poder leer los PNG (enlazo -lpng -lz)
#define cimg_use_jpeg        // por si alguna imagen viene en JPEG (-ljpeg)
#include "CImg.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>

using namespace cimg_library;

// Trabajo con imagenes de 128x128 en gris. Elegi este tamano porque deja n=16384
// y la matriz C ocupa n*n*4 ~= 1.07 GB, que entra comodo en los 16 GB de la T4
// (con 256x256 ya no cabria).
static const int W = 128;          // ancho objetivo
static const int H = 128;          // alto objetivo
static const int N = W * H;        // n = cantidad de componentes por imagen
#define TILE 16                    // lado del bloque de tiling (16x16 = 256 hilos)

// Macro para chequear el retorno de cada llamada CUDA sin repetir el if a mano.
#define CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// Lista los .png del directorio. Los ordeno por nombre para que la carga sea
// siempre la misma y los resultados sean reproducibles entre corridas.
static std::vector<std::string> listPng(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) { fprintf(stderr, "No se pudo abrir el directorio: %s\n", dir.c_str()); exit(1); }
    struct dirent* ep;
    while ((ep = readdir(dp)) != nullptr) {
        std::string name = ep->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".png")
            files.push_back(dir + "/" + name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

// Carga las imagenes ya aplanadas en h_data (m filas de N valores). Como las
// imagenes de DIV2K tienen tamanos distintos, las redimensiono todas a W x H
// para poder armar una matriz uniforme. Devuelve cuantas alcance a cargar.
static int loadDataset(const std::string& dir, int maxImages, float* h_data) {
    std::vector<std::string> files = listPng(dir);
    int m = std::min((int)files.size(), maxImages);
    if (m == 0) { fprintf(stderr, "No se encontraron .png en %s\n", dir.c_str()); exit(1); }

    for (int k = 0; k < m; k++) {
        CImg<unsigned char> img(files[k].c_str());
        img.resize(W, H, 1, img.spectrum(), 3);          // 3 = interpolacion lineal
        float* row = h_data + (size_t)k * N;
        if (img.spectrum() >= 3) {                       // si es color, paso a gris
            cimg_forXY(img, x, y) {                      // con la luminancia estandar
                float r = img(x, y, 0, 0), g = img(x, y, 0, 1), b = img(x, y, 0, 2);
                row[y * W + x] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        } else {                                         // si ya viene en gris, la copio
            cimg_forXY(img, x, y) row[y * W + x] = (float)img(x, y, 0, 0);
        }
    }
    return m;
}

// Kernel 1 - vector promedio. Pongo un hilo por componente j y ese hilo recorre
// las m imagenes sumando. Asi hilos vecinos (j, j+1) leen posiciones contiguas
// data[k*N+j], que es justo lo que necesito para que los accesos sean coalescentes.
__global__ void meanKernel(const float* __restrict__ data, float* __restrict__ mean,
                           int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float s = 0.0f;
    for (int k = 0; k < m; k++) s += data[(size_t)k * n + j];
    mean[j] = s / m;
}

// Kernel 2 - centrado. Le resto el promedio a cada imagen sobre el mismo arreglo
// (in-place) para no gastar memoria extra. Uso un grid 2D recorriendo (componente, imagen).
__global__ void centerKernel(float* __restrict__ data, const float* __restrict__ mean,
                             int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // componente
    int k = blockIdx.y * blockDim.y + threadIdx.y;   // imagen
    if (j >= n || k >= m) return;
    data[(size_t)k * n + j] -= mean[j];
}

// Kernel 3 - covarianza con tiling. Lo que necesito es C = (1/m) * Vbar^T * Vbar,
// que es una multiplicacion de matrices donde la dimension interna es k (las m
// imagenes). En vez de leer la memoria global una y otra vez, cada bloque carga
// un "tile" de la entrada en memoria compartida y lo reutiliza entre sus hilos,
// que es lo que de verdad acelera el producto. As guarda un tile de Vbar^T y Bs
// uno de Vbar; cada hilo acumula un elemento C[row][col].
__global__ void covTiledKernel(const float* __restrict__ V, float* __restrict__ C,
                               int m, int n) {
    __shared__ float As[TILE][TILE];   // As[ty][tx] = V[k][row]  (viene de Vbar^T)
    __shared__ float Bs[TILE][TILE];   // Bs[ty][tx] = V[k][col]  (viene de Vbar)

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;  // fila de salida (componente i)
    int col = blockIdx.x * TILE + tx;  // columna de salida (componente j)

    float acc = 0.0f;
    int numTiles = (m + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int kA = t * TILE + tx;        // indice k que carga este hilo en As
        int kB = t * TILE + ty;        // indice k que carga este hilo en Bs
        // Si me paso del borde (m no es multiplo de TILE) cargo 0 para no sumar basura.
        As[ty][tx] = (row < n && kA < m) ? V[(size_t)kA * n + row] : 0.0f;
        Bs[ty][tx] = (col < n && kB < m) ? V[(size_t)kB * n + col] : 0.0f;
        __syncthreads();               // espero a que el tile este completo
        #pragma unroll
        for (int e = 0; e < TILE; e++) acc += As[ty][e] * Bs[e][tx];
        __syncthreads();               // espero antes de pisar el tile en la siguiente vuelta
    }
    if (row < n && col < n) C[(size_t)row * n + col] = acc / m;
}

// =============================================================================
int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "DIV2K_valid_LR_bicubic/X4";
    int maxImages   = (argc > 2) ? atoi(argv[2]) : 100;

    // Aca uso memoria normal (paginada) a proposito: esta version hace la copia
    // sincronica, no necesito memoria anclada todavia (eso lo dejo para el Exp 2).
    float* h_data = (float*)malloc((size_t)maxImages * N * sizeof(float));
    int m = loadDataset(dir, maxImages, h_data);
    printf("[Exp1] m=%d imagenes, n=%d, matriz C = %dx%d (%.2f GB)\n",
           m, N, N, N, (double)N * N * sizeof(float) / 1e9);

    // Reservo en la GPU: el dataset, el vector promedio y la matriz de covarianza.
    float *d_data, *d_mean, *d_cov;
    CHECK(cudaMalloc(&d_data, (size_t)m * N * sizeof(float)));
    CHECK(cudaMalloc(&d_mean, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_cov,  (size_t)N * N * sizeof(float)));

    // Uso eventos CUDA para cronometrar cada etapa por separado.
    cudaEvent_t e0, e1, e2, e3;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventCreate(&e2)); CHECK(cudaEventCreate(&e3));

    // (a) Copia del dataset hacia la GPU, sincronica.
    CHECK(cudaEventRecord(e0));
    CHECK(cudaMemcpy(d_data, h_data, (size_t)m * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(e1));

    // (b) Los tres kernels en orden: promedio, centrado y covarianza.
    int blk = 256;
    meanKernel<<<(N + blk - 1) / blk, blk>>>(d_data, d_mean, m, N);

    dim3 cBlock(64, 4);
    dim3 cGrid((N + cBlock.x - 1) / cBlock.x, (m + cBlock.y - 1) / cBlock.y);
    centerKernel<<<cGrid, cBlock>>>(d_data, d_mean, m, N);

    dim3 tBlock(TILE, TILE);
    dim3 tGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    covTiledKernel<<<tGrid, tBlock>>>(d_data, d_cov, m, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e2));

    // (c) Devuelvo la matriz C al host. Ojo que C es lo mas pesado (1 GB), asi que
    //     espero que esta copia sea la mas lenta de las tres.
    float* h_cov = (float*)malloc((size_t)N * N * sizeof(float));
    CHECK(cudaMemcpy(h_cov, d_cov, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(e3));
    CHECK(cudaEventSynchronize(e3));

    float tH2D, tCompute, tD2H;
    CHECK(cudaEventElapsedTime(&tH2D,     e0, e1));
    CHECK(cudaEventElapsedTime(&tCompute, e1, e2));
    CHECK(cudaEventElapsedTime(&tD2H,     e2, e3));
    printf("[Exp1] Copia H2D dataset : %8.3f ms\n", tH2D);
    printf("[Exp1] Computo kernels   : %8.3f ms\n", tCompute);
    printf("[Exp1] Copia D2H matriz C: %8.3f ms\n", tD2H);
    printf("[Exp1] TOTAL             : %8.3f ms\n", tH2D + tCompute + tD2H);

    // Para verificar que el resultado esta bien saco la traza y la suma total.
    // Acumulo en double porque sumar millones de floats pierde precision si no.
    // Estos mismos numeros me sirven para comparar contra el Experimento 2.
    double trace = 0.0, checksum = 0.0;
    for (int i = 0; i < N; i++) trace += h_cov[(size_t)i * N + i];
    for (size_t i = 0; i < (size_t)N * N; i++) checksum += h_cov[i];
    printf("[Exp1] traza(C)=%.6e  C[0][0]=%.6f  C[0][1]=%.6f  checksum=%.6e\n",
           trace, h_cov[0], h_cov[1], checksum);

    free(h_data); free(h_cov);
    cudaFree(d_data); cudaFree(d_mean); cudaFree(d_cov);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    return 0;
}
