// =============================================================================
//  Tarea 2 - Introduccion a la Computacion Paralela
//  Experimento 2: version optimizada con CUDA Streams (pinned memory + async).
//
//  La idea es no cargar todo de golpe sino partir las imagenes en batches y
//  procesarlas en varios streams a la vez, para que mientras la GPU calcula un
//  batch, se vaya copiando el siguiente. El detalle es que para centrar necesito
//  el promedio de TODO el conjunto, lo que obligaria a una pasada extra. Para
//  evitarlo, en vez de centrar, desarrollo la covarianza con esta identidad:
//
//     C = (1/m) sum_k (v-mu)(v-mu)^T = (1/m) sum_k v v^T  -  mu mu^T
//
//  De esta forma cada batch puede aportar por su cuenta dos sumas parciales:
//     - sum de v       -> para el promedio mu      (meanSumKernel)
//     - sum de v v^T    -> momento de 2do orden     (covAccumKernel, con atomicAdd)
//  y recien al final aplico la correccion -mu mu^T (finalizeKernel). Como la suma
//  es asociativa, da exactamente la misma C que el Experimento 1, asi que comparo
//  la traza y el checksum de ambos para asegurarme de que esta bien.
//
//  Para poder estudiar bien el efecto de los streams dejo dos modos:
//     - mode=resize : una muestra por imagen redimensionada a P x P. Con P=128 el
//                     computo manda y la transferencia es despreciable (compute-bound).
//     - mode=patch  : troceo cada imagen en parches P x P y junto muchas muestras.
//                     Con P=32 y miles de parches la copia ya pesa (transfer-bound).
//
//  Compilar:  nvcc -O3 -o exp2 experimento2.cu -lpng -ljpeg -lz -lpthread -I.
//  Ejecutar:  ./exp2 <dir> <numMuestras> <S> <batchSize> [P=128] [mode=resize|patch]
//                    [reps=3] [timelineCSV]
//    compute-bound :  ./exp2 DIV2K_valid_LR_bicubic/X4 100   8 10  128 resize 3
//    transfer-bound:  ./exp2 DIV2K_valid_LR_bicubic/X4 15000 8 100 32  patch  3
//    timeline      :  ./exp2 DIV2K_valid_LR_bicubic/X4 2000  8 100 32  patch  1 tl.csv
// =============================================================================

#define cimg_display 0
#define cimg_use_png
#define cimg_use_jpeg
#include "CImg.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>

using namespace cimg_library;

#define TILE 16

#define CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ----------------------------- Lectura de imagenes ---------------------------
// Mismo listado ordenado que en el Exp 1 para tener corridas reproducibles.
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

// Paso una imagen a gris (luminancia). La saco a una funcion aparte porque la
// reuso en los dos modos de carga.
static void toGray(const CImg<unsigned char>& img, std::vector<float>& gray) {
    int wi = img.width(), hi = img.height();
    gray.resize((size_t)wi * hi);
    if (img.spectrum() >= 3) {
        cimg_forXY(img, x, y)
            gray[(size_t)y * wi + x] = 0.299f * img(x, y, 0, 0)
                                     + 0.587f * img(x, y, 0, 1)
                                     + 0.114f * img(x, y, 0, 2);
    } else {
        cimg_forXY(img, x, y) gray[(size_t)y * wi + x] = (float)img(x, y, 0, 0);
    }
}

// Modo resize: igual que el Exp 1, una muestra por imagen llevada a P x P.
static int loadResize(const std::string& dir, int maxSamples, int P, float* h_data) {
    std::vector<std::string> files = listPng(dir);
    int m = std::min((int)files.size(), maxSamples);
    if (m == 0) { fprintf(stderr, "No se encontraron .png en %s\n", dir.c_str()); exit(1); }
    int n = P * P;
    std::vector<float> gray;
    for (int k = 0; k < m; k++) {
        CImg<unsigned char> img(files[k].c_str());
        img.resize(P, P, 1, img.spectrum(), 3);
        toGray(img, gray);
        memcpy(h_data + (size_t)k * n, gray.data(), (size_t)n * sizeof(float));
    }
    return m;
}

// Modo patch: en vez de una muestra por imagen, troceo cada imagen en parches
// P x P sin solapar. Asi consigo miles de muestras chicas a partir de las 100
// imagenes, que es lo que necesito para llevar el problema al caso transfer-bound.
static int loadPatches(const std::string& dir, int maxSamples, int P, float* h_data) {
    std::vector<std::string> files = listPng(dir);
    if (files.empty()) { fprintf(stderr, "No se encontraron .png en %s\n", dir.c_str()); exit(1); }
    int n = P * P;
    int collected = 0;
    std::vector<float> gray;
    for (size_t f = 0; f < files.size() && collected < maxSamples; f++) {
        CImg<unsigned char> img(files[f].c_str());
        int wi = img.width(), hi = img.height();
        toGray(img, gray);
        for (int py = 0; py + P <= hi && collected < maxSamples; py += P) {
            for (int px = 0; px + P <= wi && collected < maxSamples; px += P) {
                float* row = h_data + (size_t)collected * n;
                for (int yy = 0; yy < P; yy++)
                    for (int xx = 0; xx < P; xx++)
                        row[yy * P + xx] = gray[(size_t)(py + yy) * wi + (px + xx)];
                collected++;
            }
        }
    }
    if (collected == 0) { fprintf(stderr, "Imagenes mas chicas que P=%d\n", P); exit(1); }
    return collected;
}

// ------------------------------- Kernels -------------------------------------

// Suma parcial del promedio: cada hilo suma su componente j sobre las imagenes
// del batch. Como distintos streams escriben sobre el mismo meanSum a la vez,
// uso atomicAdd para que no se pisen.
__global__ void meanSumKernel(const float* __restrict__ V, float* __restrict__ meanSum,
                              int bs, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float s = 0.0f;
    for (int k = 0; k < bs; k++) s += V[(size_t)k * n + j];
    atomicAdd(&meanSum[j], s);
}

// Aporte de un batch a la covarianza: calcula V_batch^T * V_batch con el mismo
// tiling del Exp 1, pero como cada batch suma su parte a la C global y pueden
// estar corriendo varios streams en paralelo, el resultado lo acumulo con atomicAdd.
__global__ void covAccumKernel(const float* __restrict__ V, float* __restrict__ C,
                               int bs, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    int numTiles = (bs + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int kA = t * TILE + tx;
        int kB = t * TILE + ty;
        As[ty][tx] = (row < n && kA < bs) ? V[(size_t)kA * n + row] : 0.0f;
        Bs[ty][tx] = (col < n && kB < bs) ? V[(size_t)kB * n + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < TILE; e++) acc += As[ty][e] * Bs[e][tx];
        __syncthreads();
    }
    if (row < n && col < n) atomicAdd(&C[(size_t)row * n + col], acc);
}

// Paso final: aplico la correccion de la identidad. Lo que tengo acumulado en C
// es sum(v v^T), asi que divido por m y le resto mu*mu^T para obtener la covarianza.
__global__ void finalizeKernel(float* __restrict__ C, const float* __restrict__ meanSum,
                               int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float mui = meanSum[row] / m;
    float muj = meanSum[col] / m;
    C[(size_t)row * n + col] = C[(size_t)row * n + col] / m - mui * muj;
}

// =============================================================================
int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "DIV2K_valid_LR_bicubic/X4";
    int maxSamples  = (argc > 2) ? atoi(argv[2]) : 100;
    int S           = (argc > 3) ? atoi(argv[3]) : 1;            // cantidad de streams
    int batchSize   = (argc > 4) ? atoi(argv[4]) : 10;           // imagenes por batch
    int P           = (argc > 5) ? atoi(argv[5]) : 128;          // lado de imagen/parche
    std::string mode = (argc > 6) ? argv[6] : "resize";          // "resize" o "patch"
    int reps        = (argc > 7) ? atoi(argv[7]) : 3;            // repeticiones para promediar
    std::string tlFile = (argc > 8) ? argv[8] : "";              // si lo paso, vuelco el timeline
    if (S < 1) S = 1;
    int n = P * P;
    bool timeline = !tlFile.empty();
    if (timeline) reps = 1;             // para el timeline me basta una corrida

    // Memoria del host ANCLADA (pinned). Es indispensable: solo con pinned el
    // cudaMemcpyAsync puede solaparse de verdad con el computo de otro stream.
    float* h_data;
    CHECK(cudaMallocHost(&h_data, (size_t)maxSamples * n * sizeof(float)));
    int m = (mode == "patch") ? loadPatches(dir, maxSamples, P, h_data)
                              : loadResize(dir, maxSamples, P, h_data);
    int numBatches = (m + batchSize - 1) / batchSize;
    printf("[Exp2] modo=%s  m=%d  P=%d (n=%d)  S=%d  batch=%d  (%d batches)  "
           "C=%.4f GB  dataset=%.4f GB\n",
           mode.c_str(), m, P, n, S, batchSize, numBatches,
           (double)n * n * sizeof(float) / 1e9,
           (double)m * n * sizeof(float) / 1e9);

    // C y el acumulador del promedio viven en la GPU todo el rato; los batches van
    // entrando por encima. Notar que aca solo guardo C (n x n) y S buffers chicos,
    // nunca el dataset entero, asi que esto escala aunque haya muchisimas muestras.
    float *d_cov, *d_meanSum;
    CHECK(cudaMalloc(&d_cov,     (size_t)n * n * sizeof(float)));
    CHECK(cudaMalloc(&d_meanSum, (size_t)n * sizeof(float)));

    // Un buffer y un stream por cada stream logico. Reusar el buffer del stream s
    // entre sus batches es seguro porque dentro de un stream todo va en orden.
    std::vector<float*>       d_batch(S);
    std::vector<cudaStream_t> stream(S);
    for (int s = 0; s < S; s++) {
        CHECK(cudaMalloc(&d_batch[s], (size_t)batchSize * n * sizeof(float)));
        CHECK(cudaStreamCreate(&stream[s]));
    }

    cudaEvent_t eStart, eEnd;
    CHECK(cudaEventCreate(&eStart)); CHECK(cudaEventCreate(&eEnd));

    // Si pido timeline, creo eventos por batch para marcar inicio/fin de copia y
    // fin de computo, y despues reconstruyo el diagrama de Gantt con esos tiempos.
    std::vector<cudaEvent_t> evCopyIni, evCopyFin, evCompFin;
    if (timeline) {
        evCopyIni.resize(numBatches); evCopyFin.resize(numBatches); evCompFin.resize(numBatches);
        for (int b = 0; b < numBatches; b++) {
            CHECK(cudaEventCreate(&evCopyIni[b]));
            CHECK(cudaEventCreate(&evCopyFin[b]));
            CHECK(cudaEventCreate(&evCompFin[b]));
        }
    }

    dim3 tBlock(TILE, TILE);
    dim3 tGrid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    int  blk = 256;

    // Repito la medicion varias veces y despues me quedo con el minimo, asi saco
    // el ruido de las primeras corridas (warm-up) y los numeros quedan estables.
    std::vector<float> tiempos;
    for (int rep = 0; rep < reps; rep++) {
        CHECK(cudaMemset(d_cov,     0, (size_t)n * n * sizeof(float)));   // arranco C en 0
        CHECK(cudaMemset(d_meanSum, 0, (size_t)n * sizeof(float)));

        CHECK(cudaEventRecord(eStart));
        // Reparto los batches entre los streams en round-robin. La gracia es que la
        // copia del batch b (stream s) se solapa con el computo del batch anterior
        // que quedo en otro stream.
        for (int b = 0; b < numBatches; b++) {
            int s   = b % S;
            int off = b * batchSize;
            int bs  = std::min(batchSize, m - off);             // el ultimo batch puede ser mas chico
            if (timeline) CHECK(cudaEventRecord(evCopyIni[b], stream[s]));
            CHECK(cudaMemcpyAsync(d_batch[s], h_data + (size_t)off * n,
                                  (size_t)bs * n * sizeof(float),
                                  cudaMemcpyHostToDevice, stream[s]));
            if (timeline) CHECK(cudaEventRecord(evCopyFin[b], stream[s]));
            meanSumKernel<<<(n + blk - 1) / blk, blk, 0, stream[s]>>>(d_batch[s], d_meanSum, bs, n);
            covAccumKernel<<<tGrid, tBlock, 0, stream[s]>>>(d_batch[s], d_cov, bs, n);
            if (timeline) CHECK(cudaEventRecord(evCompFin[b], stream[s]));
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());                         // espero a que terminen todos los streams

        dim3 fBlock(16, 16);
        dim3 fGrid((n + 15) / 16, (n + 15) / 16);
        finalizeKernel<<<fGrid, fBlock>>>(d_cov, d_meanSum, m, n);
        CHECK(cudaEventRecord(eEnd));
        CHECK(cudaEventSynchronize(eEnd));

        float ms; CHECK(cudaEventElapsedTime(&ms, eStart, eEnd));
        tiempos.push_back(ms);
    }

    // Reporto el minimo (y la mediana de referencia) sobre las repeticiones.
    std::sort(tiempos.begin(), tiempos.end());
    float tMin = tiempos.front();
    float tMed = tiempos[tiempos.size() / 2];
    printf("[Exp2] TIEMPO (S=%d) min=%.3f ms  mediana=%.3f ms  (reps=%d)\n",
           S, tMin, tMed, reps);

    // Si pedi timeline, paso los eventos a milisegundos relativos al inicio y los
    // escribo a CSV para graficar despues el solapamiento en Python.
    if (timeline) {
        FILE* f = fopen(tlFile.c_str(), "w");
        fprintf(f, "batch,stream,copy_ini_ms,copy_fin_ms,comp_fin_ms\n");
        for (int b = 0; b < numBatches; b++) {
            float ci, cf, pf;
            CHECK(cudaEventElapsedTime(&ci, eStart, evCopyIni[b]));
            CHECK(cudaEventElapsedTime(&cf, eStart, evCopyFin[b]));
            CHECK(cudaEventElapsedTime(&pf, eStart, evCompFin[b]));
            fprintf(f, "%d,%d,%.4f,%.4f,%.4f\n", b, b % S, ci, cf, pf);
        }
        fclose(f);
        printf("[Exp2] Timeline escrito en %s\n", tlFile.c_str());
    }

    // Bajo C al host y saco traza y checksum. Tienen que dar lo mismo que el Exp 1.
    float* h_cov = (float*)malloc((size_t)n * n * sizeof(float));
    CHECK(cudaMemcpy(h_cov, d_cov, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    double trace = 0.0, checksum = 0.0;
    for (int i = 0; i < n; i++) trace += h_cov[(size_t)i * n + i];
    for (size_t i = 0; i < (size_t)n * n; i++) checksum += h_cov[i];
    printf("[Exp2] traza(C)=%.6e  C[0][0]=%.6f  checksum=%.6e\n", trace, h_cov[0], checksum);

    // Libero todo.
    for (int s = 0; s < S; s++) { cudaFree(d_batch[s]); cudaStreamDestroy(stream[s]); }
    if (timeline) for (int b = 0; b < numBatches; b++) {
        cudaEventDestroy(evCopyIni[b]); cudaEventDestroy(evCopyFin[b]); cudaEventDestroy(evCompFin[b]);
    }
    free(h_cov); cudaFreeHost(h_data);
    cudaFree(d_cov); cudaFree(d_meanSum);
    cudaEventDestroy(eStart); cudaEventDestroy(eEnd);
    return 0;
}
