#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/*Make a neural net that approximates the function 1 + sin(pix/4) from -2 to 2 using a 1-2-1 network,
this means that there is 1 hidden layer of 2 neurons, and one output and input neuron.
*/


int main(){
    //We generate the random numbers 
    int Ntest=50;
    double xx[Ntest];
    double yy[Ntest];
    // make xx
    for(int i=0;i<Ntest;i++){
        //fill with random numbers from -2 to 2
        xx[i] = 4.0*rand()/RAND_MAX - 2.0;
    }
    // make yy
    for(int i=0;i<Ntest;i++){
        //make the theoretical function
        yy[i] = 1.0 + sin(M_PI*xx[i]/4.0);
    }

    // activation function
    double f(double z){
        return 1.0/(1.0+exp(-z));
    }

    // initial weights and biases distributed uniformly from -1 to 1
    double w1[2][1];
    double w2[2][1];
    double b1[2][1];
    double b2[1][1];

    // fill with random numbers from -1 to 1
    for(int i=0;i<2;i++){
        for(int j=0;j<1;j++){
            w1[i][j] = 2.0*rand()/RAND_MAX - 1.0;
            w2[i][j] = 2.0*rand()/RAND_MAX - 1.0;
        }
    }

    //Epocas
    int Ne = 50;
    //Learning rate
    double gam = 0.1;
    double x0 = 0;
    double y1_1= 0;
    double y1_2= 0;
    double n1_1= 0;
    double n1_2= 0;
    double n2_1 = 0;
    double n2_2 = 0;
    double y2_1 = 0;
    double y2_2 = 0;

    for (int epoca = 1; epoca<Ne; epoca++){
        for ( int prueba = 1; prueba<Ntest; prueba++){
            //Forward propagation
            x0=xx[prueba];
            y1_1 = w1[0][0]*x0 + b1[0][0];
            y1_2 = w1[1][0]*x0 + b1[1][0];
            y2_1 = w2[0][0]*y1 + b2[0][0];
            y2_2 = w2[1][0]*y1 + b2[0][0];
            n1_1 = f(y1_1);
            n1_2 = f(y1_2);
            n2_1 = f(y2_1);
            n2_2 = f(y2_2);


            //Error and back propagation

            //Update weights and biases


        }
    }



    return 0;
}