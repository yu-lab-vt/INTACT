#include <stdio.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

const int INFINITY = std::numeric_limits<int>::max();

float square(float x) {
    return x*x;
}

/*
 * Set a set of values in f to nan (-1) 
 */
void toNan(float *f, 
        const size_t n, 
        const size_t stride)
{
    for (size_t i = 0; i < n; i++) {
        f[i * stride] = -1;
    }
}

/* dt of 1d function using squared distance */
void dt(int *p, 
        float *f, 
        size_t n, 
        float *d, 
        const size_t stride,
        float *t, 
        const size_t t_n) 
{
    
    std::vector<int> v(n);
    std::vector<float> z(n+1);
    int k = 0;
    v[0] = 0;
    z[0] = -INFINITY;
    z[1] = +INFINITY;
    float s;
    for (size_t q = 1; q < n; q++) {
        s  = (f[q] - f[v[k]])/(2*p[q]-2*p[v[k]]) 
            + (float)(p[q] + p[v[k]])/2;
        
        while (s <= z[k]) {
            k--;
            s  = (f[q] - f[v[k]])/(2*p[q]-2*p[v[k]]) 
                + (float)(p[q] + p[v[k]])/2;
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INFINITY;
    }
    k = 0;
    for (size_t q = 0; q < t_n; q++) {
        while (z[k+1] < t[q])
            k++;
        d[q * stride] = square(t[q] - p[v[k]]) + f[v[k]];
    }
}

/* dt of 1d function using squared distance with no function on p */
void dt_noF(int *p, 
        size_t n, 
        float *d, 
        const size_t stride, 
        float *t, 
        const size_t t_n) 
{
    std::vector<int> v(n);
    std::vector<float> z(n+1);
    int k = 0;
    v[0] = 0;
    z[0] = -INFINITY;
    z[1] = +INFINITY;
    float s;
    for (size_t q = 1; q < n; q++) {
        s  = (float)(p[q] + p[v[k]])/2;
        while (s <= z[k]) {
            k--;
            s  = (float)(p[q] + p[v[k]])/2;
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INFINITY;
    }
    k = 0;
    for (size_t q = 0; q < t_n; q++) {
        while (z[k+1] < t[q])
            k++;
        d[q * stride] = square(t[q] - p[v[k]]);
    }
}

/* dt of 1d faster function using squared distance with no function on p */
void dt_noF_faster(int *p, 
        size_t n, 
        float *d, 
        const size_t stride, 
        float *t, 
        const size_t t_n) 
{
    int k = 0;
    for (size_t q = 0; q < t_n; q++) {
        if (t[q] <= p[0])
            d[q * stride] = square(t[q] - p[0]);
        else if (t[q] >= p[n-1])
            d[q * stride] = square(t[q] - p[n-1]);
        else{
            while (p[k+1] < t[q])
                k++;
            if ( (t[q] - p[k]) > (p[k+1] - t[q]) )
                d[q * stride] = square(t[q] - p[k+1]);
            else
                d[q * stride] = square(t[q] - p[k]);
        }
    }
}

/* 3D Euclidean Distance Transform - main function */
extern "C" {
    void edt_3d(const unsigned char* ref_cell,  // Use unsigned char for bool
               const int* s1_dims,             // [y, x, z] dimensions of ref_cell
               int s1_ndims,                   // number of dimensions (should be 3)
               const unsigned char* mov_cell,  // Use unsigned char for bool
               const int* s2_dims,             // [y, x, z] dimensions of mov_cell
               int s2_ndims,                   // number of dimensions (should be 3)
               const float* shift,            // [y, x, z] shifts
               float* output)                 // output array (s2_y * s2_x * s2_z)
    {
        // Validate input dimensions
        if (s1_ndims != 3 || s2_ndims != 3) {
            // Handle error - you could throw an exception or return error code
            return;
        }
        
        int s1_yxz[3], s2_yxz[3];
        if (s1_ndims == 3) {
            s1_yxz[0] = s1_dims[0];
            s1_yxz[1] = s1_dims[1];
            s1_yxz[2] = s1_dims[2];
        } else {
            // Handle 2D case if needed
            s1_yxz[0] = s1_dims[0];
            s1_yxz[1] = s1_dims[1];
            s1_yxz[2] = 1;
        }
        
        if (s2_ndims == 3) {
            s2_yxz[0] = s2_dims[0];
            s2_yxz[1] = s2_dims[1];
            s2_yxz[2] = s2_dims[2];
        } else {
            // Handle 2D case if needed
            s2_yxz[0] = s2_dims[0];
            s2_yxz[1] = s2_dims[1];
            s2_yxz[2] = 1;
        }
        
        /***distance transform along z-direction***/
        float *d2map_z = new float[s1_yxz[0] * s1_yxz[1] * s2_yxz[2]];
        
        std::vector<float> target_idx(s2_yxz[0] + s2_yxz[1] + s2_yxz[2]);
        
        for (int i=0; i<s2_yxz[2]; i++){
            target_idx[i] = i + shift[0];
        }
        
        std::vector<int> parabolas_mu(s1_yxz[0] + s1_yxz[1] + s1_yxz[2]);
        
        const size_t numEle_s1xy = s1_yxz[0] * s1_yxz[1];
        for (int y = 0; y < s1_yxz[0]; y++){
            for (int x = 0; x < s1_yxz[1]; x++){
                size_t mu_cnt = 0;
                for (int z = 0; z < s1_yxz[2]; z++){
                    if (ref_cell[z*numEle_s1xy + x*s1_yxz[0] + y]){
                        parabolas_mu[mu_cnt] = z;
                        mu_cnt++;
                    }
                }
                if (mu_cnt > 0){
                    dt_noF_faster(parabolas_mu.data(), 
                                mu_cnt, 
                                d2map_z + x*s1_yxz[0] + y, 
                                numEle_s1xy, 
                                target_idx.data(), 
                                s2_yxz[2]);
                }else{
                    toNan(d2map_z + x*s1_yxz[0] + y, 
                            s2_yxz[2], 
                            numEle_s1xy);
                }
            }
        }
        
        /***distance transform along x-direction***/
        float *d2map_zx = new float[s1_yxz[0] * s2_yxz[1] * s2_yxz[2]];
        
        for (int i=0; i<s2_yxz[1]; i++)
            target_idx[i] = i + shift[2];
        
        std::vector<float> f(s1_yxz[0] + s1_yxz[1]);
        
        const size_t numEle_s1ys2x = s1_yxz[0] * s2_yxz[1];
        for (int y = 0; y < s1_yxz[0]; y++){
            for (int z = 0; z < s2_yxz[2]; z++){
                size_t mu_cnt = 0;
                for (int x = 0; x < s1_yxz[1]; x++){
                    
                    float tmp = d2map_z[z*numEle_s1xy + x*s1_yxz[0] + y];
                    if (tmp >= 0){
                        f[mu_cnt] = tmp;
                        parabolas_mu[mu_cnt] = x;
                        mu_cnt++;
                    }
                }
                if (mu_cnt > 0){
                    dt(parabolas_mu.data(), 
                            f.data(), 
                            mu_cnt, 
                            d2map_zx + z*numEle_s1ys2x + y, 
                            s1_yxz[0], 
                            target_idx.data(), 
                            s2_yxz[1]);
                }else{
                    toNan(d2map_zx + z*numEle_s1ys2x + y, 
                            s2_yxz[1], 
                            s1_yxz[0]);
                }
            }
        }
        
        /***distance transform along y-direction ==> output***/
        for (int i=0; i<s2_yxz[0]; i++)
            target_idx[i] = i + shift[1];
        
        size_t mu_cnt = 0;
        for (int i=0; i<s1_yxz[0]; i++){
            if (d2map_zx[i] >= 0){
                parabolas_mu[mu_cnt] = i;
                mu_cnt++;
            }
        }
        
        const size_t numEle_s2yx = s2_yxz[0] * s2_yxz[1];
        for (int x = 0; x < s2_yxz[1]; x++){
            for (int z = 0; z < s2_yxz[2]; z++){
                for (size_t y = 0; y < mu_cnt; y++){
                    f[y] = d2map_zx[z * numEle_s1ys2x 
                            + x * s1_yxz[0] 
                            + parabolas_mu[y]];
                }
                dt(parabolas_mu.data(), 
                        f.data(), 
                        mu_cnt,
                        output + z*numEle_s2yx + x * s2_yxz[0], 
                        1,
                        target_idx.data(), 
                        s2_yxz[0]);
            }
        }
        
        delete [] d2map_z;
        delete [] d2map_zx;
    }
}