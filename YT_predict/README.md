Es un modelo basado en Transformadores que recibe un titulo tokenizado y predice el orden de magnitud de views que tiene un video de youtube. El modelo es relativamente pequenyo con 3M de parametros, para decidir la arquitectura me base en la documentacion de Keras acerca de NLP. Esta entrenado con un dataset de https://arxiv.org/pdf/2012.10378.pdf que son 79.2M de titulos con labels de views. Lo entrene ~50 Epochs y me dio este resultado de predicciones que me parecio bastante prometedor.