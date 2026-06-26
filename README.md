# Tarea 2 - Procesamiento de imágenes en CUDA

**Introducción a la Computación Paralela**
Autores: Joaquín Ávalos y Juan Silva — Ingeniería Civil Informática

## Descripción

Cálculo de la matriz de covarianza de un conjunto de imágenes sobre GPU usando CUDA.

- `experimento1.cu` — Versión tradicional: se carga todo el dataset contiguo en la GPU con una copia sincrónica y se calcula la covarianza con tres kernels (promedio, centrado y multiplicación matricial `Vbar^T * Vbar` con *tiling* en memoria compartida).
- `experimento2.cu` — Versión optimizada con CUDA Streams: memoria anclada (*pinned*) + copias asíncronas para solapar transferencia y cómputo. La covarianza se acumula de forma incremental usando la identidad `C = (1/m) sum(v v^T) - mu mu^T`. La cantidad de streams, el tamaño de batch y el tamaño de imagen son configurables.
- `Tarea2_Colab.ipynb` — Notebook reproducible que instala dependencias, descarga el dataset, compila y ejecuta ambos experimentos y genera los gráficos en Google Colab.

El informe en PDF con la metodología, tablas, gráficos de *speedup*, diagrama de solapamiento y el análisis se entrega por separado.

## Entorno utilizado

- **GPU:** NVIDIA Tesla T4 (16 GB VRAM) — Google Colab
- **CPU:** Intel Xeon @ 2.00 GHz (2 vCPU), 12 GiB RAM
- **CUDA:** nvcc 12.8 / driver 580.82.07
- **Dataset:** DIV2K validación, LR bicubic ×4 (100 imágenes PNG)

## Dependencias y datos

Se usa la biblioteca CImg (*header-only*) para leer los PNG:

```bash
apt-get install -y libpng-dev libjpeg-dev zlib1g-dev
wget https://raw.githubusercontent.com/GreycLab/CImg/master/CImg.h
```

Descarga del dataset:

```bash
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
unzip DIV2K_valid_LR_bicubic_X4.zip      # crea DIV2K_valid_LR_bicubic/X4/
```

## Compilación

```bash
nvcc -O3 -o exp1 experimento1.cu -lpng -ljpeg -lz -lpthread -I.
nvcc -O3 -o exp2 experimento2.cu -lpng -ljpeg -lz -lpthread -I.
```

## Ejecución

**Experimento 1:**

```bash
./exp1 DIV2K_valid_LR_bicubic/X4 100
```

**Experimento 2** — argumentos: `<dir> <numMuestras> <S> <batchSize> [P] [resize|patch] [reps] [timelineCSV]`

```bash
# Régimen compute-bound (128x128, una muestra por imagen)
./exp2 DIV2K_valid_LR_bicubic/X4 100 8 10 128 resize 3

# Régimen transfer-bound (32x32, parches)
./exp2 DIV2K_valid_LR_bicubic/X4 15000 8 100 32 patch 3

# Barrido de streams para la curva de speedup
for S in 1 2 4 8 16; do ./exp2 DIV2K_valid_LR_bicubic/X4 15000 $S 100 32 patch 3; done

# Volcado del timeline para el diagrama de solapamiento
./exp2 DIV2K_valid_LR_bicubic/X4 2000 8 100 32 patch 1 tl_S8.csv
```

## Verificación de correctitud

Ambos programas imprimen `traza(C)` y `checksum`. Estos valores coinciden entre el
Experimento 1 y el Experimento 2 (salvo en el último dígito, por el orden de las sumas
atómicas), lo que confirma que las dos versiones calculan la misma matriz de covarianza.
