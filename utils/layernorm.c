#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C)
{
    float eps = 1e-5f;
    for (int b=0; b < B; b++){
        for (int t=0; t < T; t++){
            float* x = inp + b * T * C + b * C;
            float m = 0.0f;
            for(int i=0; i < C; i++){
                m += x[i];
            }
            m = m/C;
            float v = 0.0f;
            for(int i=0; i < C; i++){
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            float s = 1.0f / sqrtf(v + eps);
            float* out_bt = out + b * T * C + t * C;
            for(int i=0; i < C; i++){
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}


void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C){
    for(int b=0; b < B; b++){
        for (int t=0; t < T; t++){
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + B * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i=0; i < C; i++){
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            for (int i=0; i < C; i++){
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dbias[i] += dout_bt[i];
                dweight[i] += norm_bti * dout_bt[i];
                float dval = 0.0f;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

int check_tensor(float *a, float *b, int n, char* label){
    int ok = 1;
    printf("%s\n", label);
    for(int i = 0; i < n; i++){
        if(fabs(a[i] - b[i]) <= 1e-5){
            printf("OK ");
        } else {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    return ok;
}


int main()
{
    int B = 2;
    int T = 3;
    int C = 4;
    size_t size = sizeof(float);
    float* x = (float*)malloc(B * T * C * size);
    float* w = (float*)malloc(C * size);
    float* b = (float*)malloc(C * size);
    float* out = (float*)malloc(B * T * C * size);
    float* mean = (float*)malloc(B * T * size);
    float* rstd = (float*)malloc(B * T * size);
    float* dout = (float*)malloc(B * T * C * size);
    float* dx = (float*)malloc(B * T * C * size);
    float* dw = (float*)malloc(C * size);
    float* db = (float*)malloc(C * size);

    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL){
        printf("Error opening file \n");
        return 1;
    }
    fread(x, size, B * T * C, file);
    fread(w, size, B * T * C, file);
    fread(b, size, B * T * C, file);
    fread(mean, size, B * T * C, file);
    fread(rstd, size, B * T * C, file);
    fread(dout, size, B * T * C, file);
    fread(dx, size, B * T * C, file);
    fread(dw, size, B * T * C, file);
    fread(db, size, B * T * C, file);
    fclose(file);

    // forward pass
    float* c_out = (float*)malloc(B * T * C * size);
    float* c_mean = (float*)malloc(B * T * size);
    float* c_rstd = (float*)malloc(B * T * size);
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    float* c_dx = (float*)calloc(B*T*C, size);
    float* c_dw = (float*)calloc(B*T, size);
    float* c_db = (float*)calloc(B*T, size);
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rstd);
    free(dout);
    free(dx);
    free(dw);
    free(db);
    return 0;
}

