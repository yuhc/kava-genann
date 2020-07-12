/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015, 2016 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#define LOOKUP_SIZE 4096

#define PRINT_OUTPUT 1

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

__device__ double genann_act_sigmoid_device(const genann *ann unused, double a) {
	if (a < -45.0) return 0;
	if (a > 45.0) return 1;
	return 1.0 / (1 + exp(-a));
}

void genann_randomize(genann *ann) {
	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		double r = GENANN_RANDOM();
		/* Sets weights from -0.5 to 0.5. */
		ann->weight[i] = r - 0.5;
	}
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
	if (hidden_layers < 0) return 0;
	if (inputs < 1) return 0;
	if (outputs < 1) return 0;
	if (hidden_layers > 0 && hidden < 1) return 0;

	const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
	const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
	const int total_weights = (hidden_weights + output_weights);

	const int total_neurons = (inputs + hidden * hidden_layers + outputs);

	/* Allocate extra size for weights, outputs, and deltas. */
	const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
	genann *ann = (genann*)malloc(size);
	if (!ann) return 0;

	ann->inputs = inputs;
	ann->hidden_layers = hidden_layers;
	ann->hidden = hidden;
	ann->outputs = outputs;

	ann->total_weights = total_weights;
	ann->total_neurons = total_neurons;

	/* Set pointers. */
	ann->weight = (double*)((char*)ann + sizeof(genann));
	ann->output = ann->weight + ann->total_weights;
	ann->delta = ann->output + ann->total_neurons;

	genann_randomize(ann);

	ann->activation_hidden = genann_act_sigmoid_device;
	ann->activation_output = genann_act_sigmoid_device;

	ann->d_ann = genann_device_copy(ann);
	return ann;
}

genann *genann_read(FILE *in) {
	int inputs, hidden_layers, hidden, outputs, ret;
	ret = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);

	genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		ret = fscanf(in, " %le", ann->weight + i);
	}

	return ann;
}

/* calculate genann pointers */
void set_genann_pointers(genann *ret) {
	/* Set pointers. */
	ret->weight = (double*)((char*)ret + sizeof(genann));
	ret->output = ret->weight + ret->total_weights;
	ret->delta = ret->output + ret->total_neurons;
}

/* calculate genann pointers for the network on device memory */
__global__ void set_genann_pointers_device(genann *d_ann) {
	/* Set pointers. */
	d_ann->weight = (double*)((char*)d_ann + sizeof(genann));
	d_ann->output = d_ann->weight + d_ann->total_weights;
	d_ann->delta = d_ann->output + d_ann->total_neurons;
}

/* copy a genann struct to GPU using CUDA APIs 
 * also recalculate the pointer locations so that they point to device memory 
 */
genann *genann_device_copy(genann const *ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	genann *ret;
	cudaMalloc((void **)&ret, size);
	if (!ret) return 0;

	cudaMemcpy(ret, ann, size, cudaMemcpyHostToDevice);
	set_genann_pointers_device<<<1, 1>>>(ret);
	return ret;
}

genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = (genann *)malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    ret->d_ann = genann_device_copy(ann);

    return ret;
}

void copy_back_genann_and_print(genann const *d_ann, genann *ann) {
	const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
	cudaMemcpy((void *)ann, d_ann, size, cudaMemcpyDeviceToHost);
	set_genann_pointers(ann);
	double *w = ann->weight + (ann->hidden_layers
		? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
		: (0));
	int n = (ann->hidden_layers ? ann->hidden : ann->inputs) + 1;
	double *d = ann->delta; /* First delta. */
	if (!PRINT_OUTPUT) return;
	/*
	for (int i = 0; i < ann->outputs; i++) {
		printf("weight: %lf\n", w[i]);
	for (int i = 0; i < ann->inputs; i++) {
		printf("input: %lf\n", ann->output[i]);
	}
	double *o = ann->output + ann->inputs;
	for (int i = 0; i < ann->hidden; i++) {
		printf("hidden output: %lf\n", o[i]);
	}
	for (int i = 0; i < ann->hidden; i++) {
		printf("delta: %lf\n", d[i]);
	}
	printf("output : %lf\n", o[ann->hidden*ann->hidden_layers]);
    */
}

void genann_free(genann *ann) {
	/* The weight, output, and delta pointers go to the same buffer. */
	cudaFree(ann->d_ann);
	free(ann);
}

// first output
__device__ double *d_o;
// first delta
__device__ double *d_d;
// first delta of the next layer
__device__ double *d_dd;
// first output in the previous layer
__device__ double *d_i;
// first weight to first output delta
__device__ double *d_w;
// fisrt weight in the following layer
__device__ double *d_ww;
__device__ double *d_t;

/* calculate the addresses for hidden layers in forward run */
__global__ void calculate_address_hidden_forward(genann const *d_ann, int h) {
	d_w = d_ann->weight;
	d_o = d_ann->output + d_ann->inputs;
	d_i = d_ann->output;

	for (int i = 1; i <= h; i++) {
		d_w += d_ann->hidden * (i == 1 ? d_ann->inputs+1 : d_ann->hidden+1);
	}
	d_o += h * d_ann->hidden;
	for (int i = 1; i <= h; i++) {
		d_i += i == 1 ? d_ann->inputs : d_ann->hidden;
	}
}

/* do forward run for hidden layers */
__global__ void genann_run_hidden(genann const *d_ann, int h) {
	int j = threadIdx.x;
	int n = (h == 0 ? d_ann->inputs : d_ann->hidden);
	double sum = d_w[(n+1)*j] * -1.0;

	for (int k = 0; k < n; ++k) {
		sum += d_w[(n+1)*j+k+1] * d_i[k];
	}
	d_o[j] = genann_act_sigmoid_device(d_ann, sum);
}

/* run for the output layer */
__global__ void genann_run_output(genann const *d_ann) {
	int j = threadIdx.x;
	int n = d_ann->hidden_layers ? d_ann->hidden : d_ann->inputs;

	/* Figure output layer. */
	double sum = d_w[j * (n+1)] * -1.0;
	for (int k = 0; k < n; ++k) {
		sum += d_w[j * (n+1) + k + 1] * d_i[k];
	}
	d_o[j] = genann_act_sigmoid_device(d_ann, sum);
}

__global__ void copy_inputs_to_device(genann *d_ann, double const *inputs, int number_inputs) {
	for (int i = 0; i < number_inputs; i++) {
		d_ann->output[i] = inputs[i];
	}
}

/* internal forward run
 * returns the genann on GPU memory
 */
void genann_run_internal(genann *ann, double const *inputs) {
	genann* d_ann = ann->d_ann;
	double *d_inputs;
	cudaMalloc((void**)&d_inputs, sizeof(double) * ann->inputs);
	cudaMemcpy(d_inputs, inputs, sizeof(double) * ann->inputs, cudaMemcpyHostToDevice);
	copy_inputs_to_device << <1, 1 >> > (d_ann, d_inputs, ann->inputs);
	copy_back_genann_and_print(d_ann, ann);
	/* copy to device to run on GPU */
	//memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
	//genann *d_ann = genann_device_copy(ann);
	/* Copy the inputs to the scratch area, where we also store each d_neuron's
	* output, for consistency. This way the first layer isn't a special case. */

	int h;

	/* Figure hidden layers, if any. */
	for (h = 0; h < ann->hidden_layers; ++h) {
		calculate_address_hidden_forward << <1, 1 >> > (d_ann, h);
		genann_run_hidden << <1, ann->hidden >> > (d_ann, h);
	}

	calculate_address_hidden_forward << <1, 1 >> > (d_ann, ann->hidden_layers);
	genann_run_output << <1, ann->outputs >> > (d_ann);
}

/* external API for running genann forward */
double const *genann_run(genann const *ann, double const *inputs) {
	genann_run_internal((genann *)ann, inputs);
	copy_back_genann_and_print(ann->d_ann, (genann *)ann);
	return ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
}

__global__ void calculate_device_addresses_output_delta(genann const *d_ann, double *d_desired_outputs) {
	d_o = d_ann->output + d_ann->inputs + d_ann->hidden * d_ann->hidden_layers; /* First output. */
	d_d = d_ann->delta + d_ann->hidden * d_ann->hidden_layers; /* First delta. */
	d_t = d_desired_outputs; /* First desired output. */
}

__global__ void calculate_device_addresses_hidden_delta(genann const *d_ann, int h) {
	d_o = d_ann->output + d_ann->inputs + (h * d_ann->hidden);
	d_d = d_ann->delta + (h * d_ann->hidden);
	d_dd = d_ann->delta + ((h + 1) * d_ann->hidden);
	d_ww = d_ann->weight + ((d_ann->inputs + 1) * d_ann->hidden) + ((d_ann->hidden + 1) * d_ann->hidden * (h));
}

__global__ void calculate_device_addresses_train_outputs(genann const *d_ann) {
	/* Find first output delta. */
	d_d = d_ann->delta + d_ann->hidden * d_ann->hidden_layers;

	/* Find first weight to first output delta. */
	d_w = d_ann->weight + (d_ann->hidden_layers
		? ((d_ann->inputs + 1) * d_ann->hidden + (d_ann->hidden + 1) * d_ann->hidden * (d_ann->hidden_layers - 1))
		: (0));

	/* Find first output in previous layer. */
	d_i = d_ann->output + (d_ann->hidden_layers
		? (d_ann->inputs + (d_ann->hidden) * (d_ann->hidden_layers - 1))
		: 0);
}

/* calculate the device addresses for training hidden layer
 * h is the index of the hidden layer
 */
__global__ void calculate_device_addresses_train_hidden(genann const *d_ann, int h) {
	/* Find first delta in this layer. */
	d_d = d_ann->delta + (h * d_ann->hidden);

	/* Find first input to this layer. */
	d_i = d_ann->output + (h
		? (d_ann->inputs + d_ann->hidden * (h - 1))
		: 0);

	/* Find first weight to this layer. */
	d_w = d_ann->weight + (h
		? ((d_ann->inputs + 1) * d_ann->hidden + (d_ann->hidden + 1) * (d_ann->hidden) * (h - 1))
		: 0);
}

/* Kernel for calculating output layer deltas*/
__global__ void calculate_output_layer_deltas(genann *d_ann) {
	int i = threadIdx.x;
	d_d[i] = (d_t[i] - d_o[i]) * d_o[i] * (1.0 - d_o[i]);
}

/* calcualte the deltas of the hidden layer */
__global__ void calc_hidden_delta(genann const *d_ann, int h) {
	double delta = 0;
	int j = threadIdx.x;

	for (int k = 0; k < (h == d_ann->hidden_layers - 1 ? d_ann->outputs : d_ann->hidden); ++k) {
		const double forward_delta = d_dd[k];
		const int windex = k * (d_ann->hidden + 1) + (j + 1);
		const double forward_weight = d_ww[windex];
		delta += forward_delta * forward_weight;
	}

	d_d[j] = d_o[j] * (1.0 - d_o[j]) * delta;
	__syncthreads();
}

/* train for the weights of the output layer */
__global__ void train_output_weights(genann const *d_ann, double learning_rate) {
	int j = threadIdx.x;
	int n = (d_ann->hidden_layers ? d_ann->hidden : d_ann->inputs) + 1;
	for (int k = 0; k < n; ++k) {
		if (k == 0) {
			d_w[n*j+k] += d_d[j] * learning_rate * -1.0;
		}
		else {
			d_w[n*j + k] += d_d[j] * learning_rate * d_i[k - 1];
		}
	}
}

__global__ void train_hidden_weights(genann const *d_ann, int h, double learning_rate) {
	int j = threadIdx.x;
	int n = (h == 0 ? d_ann->inputs : d_ann->hidden) + 1;
	for (int k = 0; k < n; ++k) {
		if (k == 0) {
			d_w[n*j+k] += d_d[j] * learning_rate * -1.0;
		}
		else {
			d_w[n*j + k] += d_d[j] * learning_rate * d_i[k - 1];
		}
	}
}

/* impl of the external API genann_train */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
	genann *d_ann = ann->d_ann;

	genann_run_internal((genann *)ann, inputs);

	double *d_desired_outputs;
	cudaMalloc((void **)&d_desired_outputs, sizeof(double) * ann->outputs);
	cudaMemcpy(d_desired_outputs, desired_outputs, sizeof(double) * ann->outputs, cudaMemcpyHostToDevice);

	int h;

	/* First set the output layer deltas. */
	calculate_device_addresses_output_delta << <1, 1 >> > (d_ann, d_desired_outputs);

	calculate_output_layer_deltas<<<1, ann->outputs>>>(d_ann);

	/* Set hidden layer deltas, start on last layer and work backwards. */
	/* Note that loop is skipped in the case of hidden_layers == 0. */
	for (h = ann->hidden_layers - 1; h >= 0; --h) {
		/* Find first output and delta in this layer. */
		calculate_device_addresses_hidden_delta << <1, 1 >> > (d_ann, h);

		/*  CALL TO KERNEL FOR SETTING HIDDEN LAYER DELTAS*/
		calc_hidden_delta << <1, ann->hidden >> > (d_ann, h);
	}

	/* Train the outputs. */
	{
		/* calculate the device addresses */
		calculate_device_addresses_train_outputs << <1, 1 >> > (d_ann);
		
		/* Set output layer weights. */
		train_output_weights << <1, ann->outputs >> > (d_ann, learning_rate);
	}


	/* Train the hidden layers. */
	for (h = ann->hidden_layers - 1; h >= 0; --h) {
		calculate_device_addresses_train_hidden << <1, 1 >> > (d_ann, h);
		
		train_hidden_weights << <1, ann->hidden >> > (d_ann, h, learning_rate);
	}

	cudaFree(d_desired_outputs);
}


void genann_write(genann const *ann, FILE *out) {
	fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

	int i;
	for (i = 0; i < ann->total_weights; ++i) {
		fprintf(out, " %.20e", ann->weight[i]);
	}
}
