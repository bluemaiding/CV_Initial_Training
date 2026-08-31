#include <stdlib.h>
#include <string.h>

unsigned char* clahe_enhance(const unsigned char* src, int width, int height, int tile_size) {
    int total_pixels = width * height;
    int grid_cols = width / tile_size;
    int grid_rows = height / tile_size;
    int num_tiles = grid_rows * grid_cols;
    int tile_pixels = tile_size * tile_size;

    // 1. 分配直方图内存 (num_tiles 个 256 长度的数组)
    int (*hist)[256] = (int(*)[256])malloc(num_tiles * sizeof(int[256]));
    if (!hist) return NULL;
    memset(hist, 0, num_tiles * sizeof(int[256]));

    // 统计直方图
    for (int i = 0; i < total_pixels; i++) {
        int row = i / width;
        int col = i % width;
        int block_row = row / tile_size;
        int block_col = col / tile_size;
        int block_id = block_row * grid_cols + block_col;
        unsigned char pixel = src[i];
        hist[block_id][pixel]++;
    }

    // 2. 对比度限制 (Clip Limit)
    int clip_limit = (int)(2.0 * tile_pixels / 256) + 1;
    for (int b = 0; b < num_tiles; b++) {
        int excess = 0;
        for (int v = 0; v < 256; v++) {
            if (hist[b][v] > clip_limit) {
                excess += (hist[b][v] - clip_limit);
                hist[b][v] = clip_limit;
            }
        }
        int avg_add = excess / 256;
        int remain = excess % 256;
        for (int v = 0; v < 256; v++) {
            hist[b][v] += avg_add;
            if (remain > 0) {
                hist[b][v]++;
                remain--;
            }
        }
    }

    // 3. 计算 CDF (累积分布)
    int (*cdf)[256] = (int(*)[256])malloc(num_tiles * sizeof(int[256]));
    if (!cdf) {
        free(hist);
        return NULL;
    }
    memset(cdf, 0, num_tiles * sizeof(int[256]));

    for (int b = 0; b < num_tiles; b++) {
        int sum = 0;
        for (int v = 0; v < 256; v++) {
            sum += hist[b][v];
            cdf[b][v] = sum;
        }
    }

    // 4. 应用映射（先用精确除法，保证图像正确）
    unsigned char* dst = (unsigned char*)malloc(total_pixels);
    if (!dst) {
        free(hist);
        free(cdf);
        return NULL;
    }

    for (int i = 0; i < total_pixels; i++) {
        int row = i / width;
        int col = i % width;
        int block_row = row / tile_size;
        int block_col = col / tile_size;
        int block_id = block_row * grid_cols + block_col;

        unsigned char old = src[i];
        int cdf_val = cdf[block_id][old];
        // 精确映射，保证亮度范围正确 (0~255)
        int new_val = (cdf_val * 255) / tile_pixels;
        dst[i] = (unsigned char)new_val;
    }

    free(hist);
    free(cdf);
    return dst;
}