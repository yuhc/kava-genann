#include <stdio.h>
#include "genann.h"
#include "mnist.h"
#include <sys/time.h>
#include <time.h>

#define SIZE 28

void copy_to_double(mnist_data_ann *data_ann, mnist_data *data_raw, int cnt) {
    for (int i = 0; i < cnt; i++) {
        data_ann[i].label = (double)data_raw[i].label;
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                data_ann[i].data[j*28+k] = (double)data_raw[i].data[j][k];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    printf("GENANN example 5.\n");
    printf("Train a MNIST on CUDA GPU.\n");

    //char* labelFileName = "mnist/t10k-labels.idx1-ubyte";
    //char* imageFileName = "mnist/t10k-images.idx3-ubyte";
    char* labelFileName = "../mnist/train-labels.idx1-ubyte";
    char* imageFileName = "../mnist/train-images.idx3-ubyte";

    mnist_data *data_raw;
    mnist_data_ann *data_ann;
    unsigned int cnt;
    int ret;

    if ((ret = loadMNISTDataUpTo(imageFileName, labelFileName, &data_raw, &cnt))) {
        printf("An error occured: %d\n", ret);
    }

    data_ann = (mnist_data_ann*)malloc(sizeof(mnist_data_ann) * cnt);
    copy_to_double(data_ann, data_raw, cnt);

    printf("Loaded %d images.\n", cnt);

    int i;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    /* New network with 2 inputs, 1 hidden layer of 2 neurons, and 1 output. */
    genann *ann = genann_init(28*28, 3, 128, LABELS_SIZE);
    double * arr = (double*)malloc(sizeof(double) * LABELS_SIZE);

    clock_t begin = clock();
    /* Train on the four labeled data points many times. */
    for (i = 0; i < 5; ++i) {
        for (int j = 0; j < cnt; j++) {
            memset(arr, 0, sizeof(double) * LABELS_SIZE);
            arr[(int)data_ann[j].label] = 1;
            genann_train(ann, data_ann[j].data, arr, 0.1);
        }
    }

    int correct = 0;

    for (int j = 0; j < cnt; j++) {
        const double * out = genann_run(ann, data_ann[j].data);
        int max = 0;
        for (int k = 1; k < LABELS_SIZE; k++) {
            if (out[k] > out[max]) {
                max = k;
            }
        }
        if (max == (int)data_ann[j].label) correct++;
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time: %lf\n", time_spent);
    printf("Correct percentage: %lf\n", 1.0 * correct / cnt * 100);

    genann_free(ann);
    free(data_raw);

    gettimeofday(&end_time, NULL);
    printf("Elapsed time: %.3lf usec\n",
            (end_time.tv_sec - start_time.tv_sec) * 1000000.0 + (end_time.tv_usec - start_time.tv_usec));
    return 0;
}
