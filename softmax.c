#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>


void SoftMax(float *ptr, int width, int height){
    for(int i=0; i<width; i++){
        float res = 0.0f;
        for(int j=0; j<height; j++){
            float total=0.0f;
            for(int k=0; k<height; k++)
                total += exp(*(ptr+i*k+k));
                // printf("%f", total);
            res = exp(*(ptr+i*j+j)) / total;
            printf("softmax value is: %f\n", res);
        }
    }
}


int main(){
    srand((unsigned)time(0));
    int width, height;
    width = 5, height = 3;
    float arr[height * width];
    float *ptr;
    ptr = arr;

    for(int i=0; i<15; i++){
        *(ptr+i) = (float)(rand()%51-25) / 50.0f;
    }

    SoftMax(ptr, width, height);
    return 0;
}