#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

float z_funct(float x, float y) // input_function
{
    return exp(2*x) + 1/pow(x, 3*y) - 3*x*y;
}

float activation_sigmoid(float x, bool deriv) // sigmoid activation function
{
    if(deriv == true)
        return x*(1-x);
    return 1/(1+exp(-x));
}

float mse(float predict, float actual) // Mean Square Error
{
    return pow(actual-predict, 2)/2;
}

float error(float predict, float actual) // Error prediction
{
     return -(actual-predict);
}

float n_max(float input[], int input_amount) // Getting maximum value
{
    float maxim = input[0];
    for(int i = 1; i < input_amount; i++)
    {
        if(input[i] > maxim)
            maxim = input[i];
    }
    printf("\nNilai terbesar adalah = %f\n", maxim);
    return maxim;
}

float n_min(float input_x[], float input_y[], int input_amount) // Getting minimum value
{
    float minim = input_x[0];
    for(int i = 1; i < input_amount; i++)
    {
        if(input_x[i] < minim)
            minim = input_x[i];
    }
    for(int i = 0; i < input_amount; i++)
    {
        if(input_y[i] < minim)
            minim = input_y[i];
    }
    printf("\nNilai terkecil adalah = %f\n", minim);
    return minim;
}

float normalized(float input, float maxim, float minim)
{
    return (input - minim) / (maxim - minim);
}

float denormalized(float input, float maxim, float minim)
{
    return input * (maxim - minim) + minim;
}

void save_error_txt(float input[], int index, int epoch_number = 1)
{
  ofstream myfile("error_log.txt");
  string str;
  stringstream ss;

  ss << epoch_number;
  ss >> str;
  myfile << "\t\t\t\t\tError train logger per-" << str << " data\n\n\n";
  for(int i = 0; i < index; i++)
      if(i % epoch_number-1 == 0)
          myfile << input[i] << "\n";

  myfile.close();
}

int main()
{
    float input[100][100], weight[100][100], bias[100], lr, z[100][100], output[10][10], error_value[100], z_error[10][10];
    float output_delta[10][10], model_error, model_target[100], prev_error = 0, epoch[100000], epoch_error = 0, maxim = 0, minim = 0;
    int hidden_layer, neuron_hl, neuron_out, neuron_in, ts, a = 0, b = 0, c = 0, d = 0, e, h, i, j, k, l, m, n, train, now_location = 0, index = 0;
    char confirm = ' ', y_confirm = ' ';

    srand(time(0)); //initialize random number generator

    printf("--------------------------------------------------------------------------------------------\n");
    printf("\t\t\t\tProgram Neural Network\n");
    printf("--------------------------------------------------------------------------------------------\n");
    printf("By : Alfany Riza Mahendra - 1110181015\t\t\tFungsi input = e^2x + 1/(x^3y) - 3xy\n\n");

    printf("Masukan lebar dataset yang akan di buat : ");
    scanf("%d",&l);

    // ------------ Generate random dataset for x ,y and after function numbers --------------------
    for(i = 0; i < l ; i++)
    {
        input[i][0] = rand();
        input[i][1] = rand();
        input[i][0] = input[i][0] / 100000;
        input[i][1] = input[i][1] / 100000;
        bias[i] = input[i][0];
        error_value[i] = input[i][1];
        model_target[i] = z_funct(input[i][0], input[i][1]);
    }

    // --------------- Convert input after function to range between 0 and 1 -------------------------
    // getting maximum value
    printf("\n-------------------------------------------------------------------------------------------\n");
    printf("Rumus untuk scaling (min-max normalization) = (input - input_min) / (input_max - input_min)\n");
    printf("-------------------------------------------------------------------------------------------\n");
    maxim = n_max(model_target, l);
    minim = n_min(bias, error_value, l);

    for(i = 0; i < l; i++)
    {
        epoch[i] = model_target[i];
        model_target[i] = normalized(model_target[i], maxim, minim);
        input[i][0] = normalized(input[i][0], maxim, minim);
        input[i][1] = normalized(input[i][1], maxim, minim);
    }

    // Print input for checking
    printf("-----------------------------------------------------------------------------------------------------------------\n");
    printf("|\tNo\tinput_x,input_y\t\tinput_x,input_y_normalized\tafter_function\t\tnormalized\t|\n");
    printf("-----------------------------------------------------------------------------------------------------------------\n");
    for(i = 0; i < l ; i++)
        printf("|\t%d\t%f, %f\t    %f, %f\t\t   %f\t\t  %f\t|\n", i+1, bias[i], error_value[i], input[i][0], input[i][1], epoch[i], model_target[i]);

    printf("-----------------------------------------------------------------------------------------------------------------\n\n");

    printf("\n");

    printf("Masukkan banyak data untuk training : ");
    scanf("%d",&train);
    printf("\n");
    // Getting training_set and testing_set slice numbers
    printf("Data training yang di generate sebanyak %d dan data validasi sebanyak %d\n", train, (l-train));

    // Print input for checking
    printf("-----------------------------------------------------------------------------------------------------------------\n");
    printf("|\tNo\tTraining_set\t\t\tValidation_set\t\t\tTarget_output\t\t\t|\n");
    printf("-----------------------------------------------------------------------------------------------------------------\n");

    // Print training_set and testing_set
    for(i = 0; i < l ; i++)
    {
        epoch[i] = 0;
        if(i < train)
            printf("|\t%d\t%f, %f\t\t\t\t   \t\t  %.4f\t\t\t|\n", i+1, input[i][0], input[i][1], model_target[i]);
        else
            printf("|\t%d\t   \t\t\t\t%f, %f\t\t  %.4f\t\t\t|\n", i+1, input[i][0], input[i][1], model_target[i]);
    }
    printf("-----------------------------------------------------------------------------------------------------------------\n\n");

    // -------------------------------------------------- NN parameter input -----------------------------------------------------
    // Getting amount of neuron_out and neuron input
    neuron_out = 1;
    neuron_in = 0;
    for(int i = 0; i < l; i++)
        if(input[0][i] != 0) neuron_in++;
    printf("Jumlah neuron input = %d\nJumlah neuron output = %d\n\n", neuron_in, neuron_out);

    // Initial parameter for NN
    printf("Masukkan jumlah hidden layer : ");
    scanf("%d",&hidden_layer);
    printf("Masukkan jumlah neuron hidden layer : ");
    scanf("%d",&neuron_hl);
    printf("Masukkan learning rate : ");
    scanf("%f",&lr);
    printf("Masukkan jumlah training steps : ");
    scanf("%d",&ts);
    printf("\n");

    // ----------- Random weights and bias generator ------------------
    printf("-------------------------------------------------------------------------\n");
    printf("\t\t\tHasil generate weight dan Bias\n");
    printf("-------------------------------------------------------------------------\n\n");

    n=0;
    for(i = 0; i <= hidden_layer ;i++)
    {
        if(i == 0)
        {
            h = 0;
            for(j = 0; j < neuron_in ; j++)
            {
                for(k = 0; k < neuron_hl; k++)
                {
                    weight[n+j][k] = rand();
                    weight[n+j][k] = weight[n+j][k] / 77777;
                    printf("weight layer input, neuron %d, %d = %f\n", j, k, weight[n+j][k]);
                }
                h++;
            }
            n+=h;
            //printf("Flag = %d\n", n);
            printf("\n");
        }
        else if(i == hidden_layer)
        {
            h = 0;
            for(j = 0; j < neuron_hl ; j++)
            {
                for(k = 0; k < neuron_out; k++)
                {
                    weight[n+j][k] = rand();
                    weight[n+j][k] = weight[n+j][k] / 77777;
                    printf("weight output, neuron %d, %d = %f\n", j, k, weight[n+j][k]);
                }
                h++;
            }
            n += h;
            //printf("Flag = %d\n", n);
            printf("\n");
        }
        else
        {
            h = 0;
            for(j = 0; j < neuron_hl; j++)
            {
                for(k = 0; k < neuron_hl; k++)
                {
                    weight[n+j][k] = rand();
                    weight[n+j][k] = weight[n+j][k] / 77777;
                    printf("weight antar hidden layer %d, neuron %d, %d = %f\n", i, j, k, weight[n+j][k]);
                }
                h++;
            }
            n += h;
            bias[i-1]= rand();
            bias[i-1]= bias[i-1] / 99999;
            printf("bias hidden layer %d = %f\n", i, bias[i-1]);
            //printf("Flag = %d\n", n);
            printf("\n");
        }
    }
    printf("\n\n\n");

    // ------------------------------------------------ Running NN --------------------------------------------------------------
    for(a = 0; a < l; a++)
    {
        // ----------------------------------------------------------------------------------------------------------------------
        // -------------------------------------------- Training section --------------------------------------------------------
        if(a < train)
        {
            for(b = 0; b < ts; b++)
            {
                // ------------------------------------- Forward propagation -----------------------------------------------
                printf("---------------------------------------------------------------------------------------------------\n");
                printf("input data training ke-%d = %f, %f \t\t\t\t\ttraining steps = %d\n", a+1, input[a][0], input[a][1], b+1);
                printf("Target output = %f \t\t\t\t\t  setelah denormalisasi = %f\n",model_target[a], denormalized(model_target[a], maxim, minim));
                printf("---------------------------------------------------------------------------------------------------\n\n");
                n = 0;
                e = 0;
                model_error = 0;
                for(i = 0; i <= hidden_layer ;i++)
                {
                    if(i == 0)
                    {
                        h = 0;
                        c = 0;
                        d = 0;
                        for (j = 0; j < neuron_in; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                z[k+n][j] += input[a][j]*weight[c+j][k];
                                //printf("Hasil z ke-%d,%d adalah %f\n", k+n, j, z[k+n][j]);
                            }
                            d++;
                        }

                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_in; k++)
                            {
                                output[i][j] += z[j+n][k];
                                output[i][j] = activation_sigmoid(output[i][j] + bias[i], false);
                                //printf("output hidden layer %d neuron %d data ke-%d = %f\n", i+1, j, a, output[i][j]);
                            }
                        }
                        c += d;
                        n += d;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n", n);
                        printf("\n");
                    }
                    else if(i == hidden_layer)
                    {
                        h = 0;
                        d = 0;
                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_out; k++)
                            {
                                z[n+k][j] += output[i-1][j]*weight[c+j][k];
                                //printf("Hasil z ke-%d,%d adalah %f\n", k+n, j, z[k+n][j]);
                            }
                            d++;
                        }

                        for(j = 0; j < neuron_out; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                output[i][j] += z[j+n][k];
                                output[i][j] = activation_sigmoid(output[i][j], false);
                            }
                            printf("output hidden layer %d neuron %d data ke-%d = %f\n", i+1, j, a, output[i][j]);
                            printf("output setelah denormalized = %f\n", denormalized(output[i][j], maxim, minim));
                            error_value[j] = error(output[i][j], model_target[a]);
                            model_error += mse(output[i][j], model_target[a]);
                            e++;
                        }
                        model_error = model_error/e;
                        epoch[index] = model_error;
                        index++;
                        c += d;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n", n);
                        printf("---------------------------------------------------------------------------------------------------\n\n");
                        printf("\n");
                    }
                    else
                    {
                        d = 0;
                        h = 0;
                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                z[n+k][j] += output[i-1][j]*weight[c+j][k];
                                //printf("Hasil z ke-%d,%d adalah %f\n", k+n, j, z[k+n][j]);
                            }
                            d++;
                        }

                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                output[i][j] += z[j+n][k];
                                output[i][j] = activation_sigmoid(output[i][j] + bias[i], false);
                                //printf("output hidden layer %d neuron %d data ke-%d = %f\n", i+1, j, a, output[i][j]);
                            }
                        }
                        c += d;
                        n += d;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n", n);
                        //printf("\n");
                    }
                }
                printf("model error = %f\n\n",model_error);

                // ------------------------------------- backward propagation -----------------------------------------------
                for(i = hidden_layer; i >= 0 ;i--)
                {
                    // Update hyperparameter
                    if(i == 0)
                    {
                        d = 0;
                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                output_delta[i][j] = z_error[c + now_location - j][k]*activation_sigmoid(output[i][j], true);
                                //printf("Output delta ke-%d dari error %f dan turunan sigmoid %f = %f\n", i, z_error[c + now_location - j][k], activation_sigmoid(output[i][j], true), output_delta[i][j]);
                            }
                        }

                        for(j = 0; j < neuron_in; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                z_error[c - j][k] = output_delta[i][k]*input[a][j];
                                //printf("z_error ke-%d dari %f ,input %f dan output delta %f = %f\n", i, z[n-k][j], input[a][j], output_delta[i][k], z_error[c - j][k]);
                            }
                            d++;
                        }

                        c -= d;
                        now_location  = neuron_hl;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n\n", n);

                        for(e = 0; e <= hidden_layer ;e++)
                        {
                            if(e == 0)
                            {
                                d = 0;
                                if(a == train-1 && b == ts-1)
                                {
                                    printf("\n---------------------------------------------------------------------------------------------------\n");
                                    printf("\t\t\t\tFinal Weight and Bias Set");
                                    printf("\n---------------------------------------------------------------------------------------------------\n\n");
                                }

                                for(j = 0; j < neuron_in; j++)
                                {
                                    for(k = 0; k < neuron_hl; k++)
                                    {
                                        weight[c + j][k] -= lr*output_delta[i][k]*input[a][j];
                                        if(a == train-1 && b == ts-1)
                                            printf("weight ke-%d, %d = %f\n", c + j, k, weight[c + j][k]);
                                        z[c + j][k] = 0;
                                        //printf("New weight layer %d neuron %d, weight %d dari output delta %f dan input ke-%d,%d = %f\n", e, j, k, output_delta[i][k], a, j, weight[c + j][k]);
                                    }
                                    d++;
                                }
                                c += d;
                                //printf("Flag c = %d\n", c);
                                printf("\n");
                            }
                            else if(e == hidden_layer)
                            {
                                d = 0;
                                for(j = 0; j < neuron_hl; j++)
                                {
                                    for(k = 0; k < neuron_out; k++)
                                    {
                                        weight[c + j][k] -= lr*output_delta[i][j]*z[c + k][j];
                                        if(a == train-1 && b == ts-1)
                                            printf("weight ke-%d, %d = %f\n", c + j, k, weight[c + j][k]);
                                        //printf("New weight layer %d neuron %d, weight %d dari output delta %f dan z ke-%d,%d = %f\n", e, k, j, output_delta[i][j], c + k, j, weight[c + j][k]);
                                    }
                                    d++;

                                    for(k = 0; k < neuron_hl; k++)
                                        z[c + j][k] = 0;
                                }

                                if(a == train-1 && b == ts-1)
                                {
                                    printf("\n");
                                    for(j = 0; j < neuron_hl; j++)
                                        printf("bias ke-%d = %f\n", j+1, bias[j]);
                                }

                                c += d;
                                //printf("Flag c = %d\n", c);
                                //printf("\n");
                            }
                            else
                            {
                                d = 0;
                                for(j = 0; j < neuron_hl; j++)
                                {
                                    for(k = 0; k < neuron_hl; k++)
                                    {
                                        weight[c + j][k] -= lr*output_delta[i][j]*z[c + k][j];
                                        if(a == train-1 && b == ts-1)
                                            printf("weight ke-%d, %d = %f\n", c + j, k, weight[c + j][k]);
                                        //printf("New weight layer %d neuron %d, weight %d dari output delta %f dan z ke-%d,%d = %f\n", e, k, j, output_delta[i][j], c + k, j, weight[c + j][k]);
                                    }
                                    d++;
                                    for(k = 0; k < neuron_hl; k++)
                                        z[c + j][k] = 0;
                                }
                                c += d;
                                //printf("Flag c = %d\n", c);
                                printf("\n");
                            }
                        }
                        //printf("\n---------------------------------------------------------------------------------------------------\n\n");
                        //printf("\n");
                    }
                    // search for delta error and z error
                    else if(i == hidden_layer)
                    {
                        d = 0;
                        for(j = 0; j < neuron_out; j++)
                        {
                            output_delta[i][j] = error_value[j]*activation_sigmoid(output[i][j], true);
                            //printf("Output delta ke-%d dari error %f dan turunan sigmoid %f = %f\n", i, error_value[j], activation_sigmoid(output[i][j], true), output_delta[i][j]);
                        }

                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_out; k++)
                            {
                                z_error[c - j][k] = output_delta[i][k]*weight[n-j][k];
                                //printf("z_error ke-%d dari %f = %f\n", i, z[n-k][j], z_error[c - j][k]);
                            }
                            d++;
                        }
                        c -= d;
                        n -= d;
                        h = neuron_out;
                        now_location  = neuron_hl;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n\n", n);
                    }
                    else
                    {
                        d = 0;
                        if(h == neuron_out)
                        {
                            for(j = 0; j < neuron_hl; j++)
                            {
                                for(k = 0; k < neuron_out; k++)
                                {
                                    output_delta[i][j] = z_error[c + now_location - j][k]*activation_sigmoid(output[i][j], true);
                                    //printf("Output delta ke-%d dari error %f dan turunan sigmoid %f = %f\n", i, z_error[c + now_location - j][k], activation_sigmoid(output[i][j], true), output_delta[i][j]);
                                }
                                h = neuron_hl;
                            }
                        }

                        else
                        {
                            for(j = 0; j < neuron_hl; j++)
                            {
                                for(k = 0; k < neuron_hl; k++)
                                {
                                    output_delta[i][j] = z_error[c + now_location - j][k]*activation_sigmoid(output[i][j], true);
                                    //printf("Output delta ke-%d dari error %f dan turunan sigmoid %f = %f\n", i, z_error[c + now_location - j][k], activation_sigmoid(output[i][j], true), output_delta[i][j]);
                                }
                                h = neuron_hl;
                            }
                        }

                        for(j = 0; j < neuron_hl; j++)
                        {
                            for(k = 0; k < neuron_hl; k++)
                            {
                                z_error[c - j][k] = output_delta[i][k]*weight[n-j][k];
                                //printf("z_error ke-%d dari %f = %f\n", i, z[n-k][j], z_error[c - j][k]);
                            }
                            d++;
                        }
                        c -= d;
                        n -= d;
                        //now_location  = neuron_hl;
                        //printf("Flag c = %d\n", c);
                        //printf("Flag n = %d\n\n", n);
                    }
                }
            }
            model_error = 0;
            e = 0;
            printf("\n-------------------------------------------- Validation section --------------------------------------------------------\n");
        }

        else
        {
            // ------------------------------------------------------------------------------------------------------------------------
            // -------------------------------------------- Validation section --------------------------------------------------------
            printf("\n---------------------------------------------------------------------------------------------------------\n");
            printf("input data validasi ke-%d = %f ,%f\tTarget output = %f, setelah denormalized = %f\n", a-train+1, input[a][0], input[a][1], model_target[a], denormalized(model_target[a], maxim, minim));
            printf("---------------------------------------------------------------------------------------------------------\n\n");
            for(i = 0; i <= hidden_layer ;i++)
            {
                if(i == 0)
                {
                    c = 0;
                    d = 0;
                    for (j = 0; j < neuron_in; j++)
                    {
                        for(k = 0; k < neuron_hl; k++)
                            z[k+n][j] += input[a][j]*weight[c+j][k];
                        d++;
                    }

                    for(j = 0; j < neuron_hl; j++)
                    {
                        for(k = 0; k < neuron_in; k++)
                        {
                            output[i][j] += z[j+n][k];
                            output[i][j] = activation_sigmoid(output[i][j] + bias[i], false);
                            //printf("output hidden layer %d neuron %d data ke-%d = %f\n", i+1, j, a, output[i][j]);
                        }
                    }
                    c += d;
                    n += d;
                }
                else if(i == hidden_layer)
                {
                    d = 0;
                    for(j = 0; j < neuron_hl; j++)
                    {
                        for(k = 0; k < neuron_out; k++)
                            z[n+k][j] += output[i-1][j]*weight[c+j][k];
                        d++;
                    }

                    for(j = 0; j < neuron_out; j++)
                    {
                        for(k = 0; k < neuron_hl; k++)
                        {
                            output[i][j] += z[j+n][k];
                            output[i][j] = activation_sigmoid(output[i][j], false);
                        }
                        printf("output data validasi ke-%d = %f\n", a-train+1, output[i][j]);
                        printf("output setelah denormalized = %f\n", denormalized(output[i][j], maxim, minim));
                        model_error += mse(output[i][j], model_target[a]);
                        prev_error = mse(output[i][j], model_target[a]);
                        e++;
                    }
                    printf("model error = %f\n",prev_error);
                    printf("---------------------------------------------------------------------------------------------------------\n\n");
                    //printf("\n");
                }
                else
                {
                    d = 0;
                    h = 0;
                    for(j = 0; j < neuron_hl; j++)
                    {
                        for(k = 0; k < neuron_hl; k++)
                            z[n+k][j] += output[i-1][j]*weight[c+j][k];
                        d++;
                    }

                    for(j = 0; j < neuron_hl; j++)
                    {
                        for(k = 0; k < neuron_hl; k++)
                        {
                            output[i][j] += z[j+n][k];
                            output[i][j] = activation_sigmoid(output[i][j] + bias[i], false);
                            //printf("output hidden layer %d neuron %d data ke-%d = %f\n", i+1, j, a, output[i][j]);
                        }
                    }
                    c += d;
                    n += d;
                }
            }
        }
    }

    printf("\n------------------------------------------------------------------------------------------------------------------------\n");
    printf("\t\t\t\t\t\t Training model result \t\t\t\t\n\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n\n");
    n = 0;
    for(i = 0; i < index; i++)
    {
        if(i == 0)
            continue;

        if(ts >= 0 && ts < 501)
        {
            if(i % 10 == 0)
                printf("Error epoch per-%d = %.25f\n", i, epoch[i]);
            if(i % (index-1) == 0)
                printf("Error epoch per-%d = %.25f\n", i, epoch[i]);
        }
        else
        {
            if(i % 100 == 0)
                printf("Error epoch per-%d = %.25f\n", i, epoch[i]);
            if(i % (index-1) == 0)
                printf("Error epoch per-%d = %.25f\n", i, epoch[i]);
        }
    }

    printf("\n------------------------------------------------------------------------------------------------------------------------\n");
    printf("\t\t\t\t\t\t Batch error \t\t\t\t\n\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n\n");
    n = 0;
    for(i = 0; i < index; i++)
    {
        if(i == 0)
            continue;

        if(ts >= 0 && ts < 501)
        {
            if(i % 10 == 0)
                printf("%.25f\n", epoch[i]);
            if(i % (index-1) == 0)
                printf("%.25f\n", epoch[i]);
        }
        else
        {
            if(i % 100 == 0)
                printf("%.25f\n", epoch[i]);
            if(i % (index-1) == 0)
                printf("%.25f\n", epoch[i]);
        }
    }

    // Save every error value to txt file
    save_error_txt(epoch, index, 10);

    printf("---------------------------------------------------------------------------------------------------\n\n");

    // Print validation model result
    printf("\n\t\t\t\t Validation model result \t\t\t\t\n");
    printf("---------------------------------------------------------------------------------------------------\n\n");
    model_error = model_error/e;
    printf("\nError NN model based on validation dataset = %.15f\n", model_error);
    printf("---------------------------------------------------------------------------------------------------\n\n");
}
