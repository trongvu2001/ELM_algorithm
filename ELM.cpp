// Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// ELM structure
class ELM
{
private:
	int num_inputs;			// Number of input neurons
	int num_hidden_neurons; // Number of hidden neurons
	int num_outputs;		// Number of output neurons

public:
	double *input_weights;	// Input weights
	double *hidden_bias;	// Hidden layer bias
	double *hidden_weights; // Hidden layer weights
	double *output_weights; // Output weights
	double sigmoid(double x);
	void fit(ELM *elm, double *inputs, double *targets);
	void predict(ELM *elm, double *inputs, double *predictions);
	double rand_uniform(double min, double max);
	void init(ELM *elm, int input_size, int hidden_size, int output_size);
};

// Sigmoid activation function
double ELM::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void ELM::fit(ELM *elm, double *inputs, double *targets)
{
	// Generate random input weights and hidden layer bias
	for (int i = 0; i < elm->num_inputs * elm->num_hidden_neurons; i++)
	{
		elm->input_weights[i] = (double)rand() / RAND_MAX;
	}
	for (int i = 0; i < elm->num_hidden_neurons; i++)
	{
		elm->hidden_bias[i] = (double)rand() / RAND_MAX;
	}
	// Calculate hidden layer activations
	for (int i = 0; i < elm->num_hidden_neurons; i++)
	{
		double net = 0;
		for (int j = 0; j < elm->num_inputs; j++)
		{
			net += inputs[j] * elm->input_weights[i * elm->num_inputs + j];
		}
		elm->hidden_weights[i] = elm->sigmoid(net - elm->hidden_bias[i]);
	}
	// Calculate output weights using least squares
	for (int i = 0; i < elm->num_outputs; i++)
	{
		double error = 0;
		for (int j = 0; j < elm->num_hidden_neurons; j++)
		{
			error += elm->hidden_weights[j] * targets[j];
		}
		elm->output_weights[i] = error / elm->num_hidden_neurons;
	}
}

//// Predict method
// void ELM::predict(ELM *elm, double *inputs, double *predictions)
//{
//	// Calculate hidden layer activations
//	for (int i = 0; i < elm->num_hidden_neurons; i++)
//	{
//		double net = 0;
//		for (int j = 0; j < elm->num_inputs; j++)
//		{
//			net += inputs[j] * elm->input_weights[i * elm->num_inputs + j];
//		}
//		elm->hidden_weights[i] = elm->sigmoid(net - elm->hidden_bias[i]);
//	}
//	// Calculate output predictions
//	for (int i = 0; i < elm->num_outputs; i++)
//	{
//		double prediction = 0;
//		for (int j = 0; j < elm->num_hidden_neurons; j++)
//		{
//			prediction += elm->hidden_weights[j] * elm->output_weights[i];
//		}
//		predictions[i] = prediction;
//	}
// }

// Predict method
void ELM::predict(ELM *elm, double *inputs, double *predictions)
{
	// Calculate hidden layer activations
	for (int i = 0; i < elm->num_hidden_neurons; i++)
	{
		double net = 0;
		for (int j = 0; j < elm->num_inputs; j++)
		{
			net += inputs[j] * elm->input_weights[i * elm->num_inputs + j];
		}
		elm->hidden_weights[i] = elm->sigmoid(net - elm->hidden_bias[i]);
	}
	// Calculate output predictions
	for (int i = 0; i < elm->num_outputs; i++)
	{
		double prediction = 0;
		for (int j = 0; j < elm->num_hidden_neurons; j++)
		{
			prediction += elm->hidden_weights[j] * elm->output_weights[j];
		}
		predictions[i] = prediction;
	}
}

double ELM::rand_uniform(double min, double max)
{
	double range = max - min;
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

void ELM::init(ELM *elm, int input_size, int hidden_size, int output_size)
{
	// allocate and initialize memory for input weights
	elm->input_weights = (double *)malloc(input_size * hidden_size * sizeof(double));
	for (int i = 0; i < input_size * hidden_size; i++)
	{
		elm->input_weights[i] = rand_uniform(-1, 1);
	}

	// allocate and initialize memory for hidden bias
	elm->hidden_bias = (double *)malloc(hidden_size * sizeof(double));
	for (int i = 0; i < hidden_size; i++)
	{
		elm->hidden_bias[i] = rand_uniform(-1, 1);
	}

	// allocate and initialize memory for hidden weights
	elm->hidden_weights = (double *)malloc(hidden_size * output_size * sizeof(double));
	for (int i = 0; i < output_size * hidden_size; i++)
	{
		elm->hidden_weights[i] = rand_uniform(-1, 1);
	}

	// allocate and initialize memory for output weights
	elm->output_weights = (double *)malloc(output_size * sizeof(double));
	for (int i = 0; i < output_size; i++)
	{
		elm->output_weights[i] = rand_uniform(-1, 1);
	}
	elm->num_inputs = input_size;
	elm->num_hidden_neurons = hidden_size;
	elm->num_outputs = output_size;
}

int main()
{
	// Initialize the ELM
	ELM elm;
	elm.init(&elm, 2, 3, 1);

	// Train the ELM with some inputs and targets
	double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double targets[4][1] = {{0}, {1}, {1}, {0}};
	elm.fit(&elm, (double *)inputs, (double *)targets);

	// Make a prediction
	double new_input[2] = {1, 0};
	double prediction[1];
	elm.predict(&elm, new_input, prediction);
	printf("Prediction: %lf \n", prediction[0]);

	// Free allocated memory
	free(elm.input_weights);
	free(elm.hidden_bias);
	free(elm.hidden_weights);
	free(elm.output_weights);

	return 0;
}
