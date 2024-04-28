# Redes Neuronales Convolucionales (CNN) - Una Visión General
Las redes neuronales convolucionales (CNN, por sus siglas en inglés "Convolutional Neural Networks") son un tipo de redes neuronales profundas que se utilizan ampliamente para el procesamiento de imágenes, reconocimiento de patrones y tareas relacionadas. Aquí te dejo una visión general:

## Estructura básica
Las CNN están diseñadas para procesar datos que tienen una estructura de cuadrícula, como imágenes. Una imagen puede ser vista como una cuadrícula de píxeles en 2D o 3D (para imágenes en color).

## Como funcionan las Imagenes
Las imágenes se representan como tensores (matrices multidimensionales) donde las dimensiones corresponden a la altura, el ancho y los canales de color (por ejemplo, rojo, verde y azul en una imagen RGB). Lo que hace que se napea a un array bidimensional de píxeles, donde cada píxel tiene un valor de intensidad de color.

## La operación de convolución
La operación de convolución es el corazón de una red neuronal convolucional (CNN) y es fundamental para entender cómo estas redes aprenden a interpretar imágenes. Vamos a desglosarlo:

 $$ (x * y) (x) = \int_{-\infty}^{\infty} x(\tau) y(x - \tau) d\tau $$

### Concepto de Convolució
La convolución es una operación matemática que consiste en tomar dos funciones (en el contexto de las CNN, una imagen y un filtro) y producir una tercera función que representa la forma en que una se superpone a la otra. En términos de procesamiento de imágenes, es como si pasáramos una ventana (el filtro) sobre toda la imagen para analizar pequeñas partes de ella a la vez.
![Convolución](https://miro.medium.com/max/1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)

### Componentes de la Convolución
Imagen de Entrada: Es la matriz de píxeles que se va a analizar. Puede ser en escala de grises (2D) o en color (3D si consideramos los canales de color).
Filtro o Kernel: Es una pequeña matriz que se utiliza para detectar características específicas, como bordes, texturas o patrones. El tamaño del filtro es mucho menor que el de la imagen y puede variar según lo que quieras detectar (comúnmente 3x3, 5x5).
Mapa de Características (Feature Map): Es el resultado de aplicar el filtro a la imagen. Captura la información de cómo y dónde se detectan ciertas características en la imagen.

### Proceso de Convolución
Superposición: Colocamos el filtro sobre una parte de la imagen.
Elemento por Elemento Multiplicación: Multiplicamos cada elemento del filtro con el elemento correspondiente de la imagen bajo el filtro.

Suma: Sumamos todos los productos obtenidos en el paso anterior. Este resultado es un único valor en el nuevo mapa de características.

Deslizamiento: Desplazamos el filtro un paso hacia la derecha (o hacia abajo) y repetimos el proceso hasta que hemos cubierto toda la imagen.

### Stride y Padding

* Stride: Define cuántos píxeles desplazamos el filtro en cada paso. Un stride más grande resulta en un mapa de características más pequeño.

* Padding: A veces añadimos filas o columnas de ceros alrededor de la imagen de entrada para permitir que el filtro se aplique a los bordes de la imagen. Esto ayuda a controlar el tamaño del mapa de características y a asegurar que cada píxel influya en el resultado.

### Resultado de la Convolución
La convolución transforma una imagen grande en una serie de mapas de características más pequeños que resumen las presencias de características específicas en diferentes regiones de la imagen. Esto es útil porque en las capas subsiguientes, la red puede empezar a reconocer combinaciones de estas características simples para formar conceptos más complejos (por ejemplo, un conjunto de líneas puede formar una forma).

La belleza de la convolución radica en su capacidad para preservar la relación espacial entre píxeles y ser eficiente en términos de aprendizaje de características relevantes, lo que reduce la cantidad de parámetros que la red necesita aprender en comparación con una red totalmente conectada. Esto hace a las CNN extremadamente poderosas para tareas relacionadas con la visión por computadora.

## La Capa Relu

La función de activación ReLU, o "Rectified Linear Unit", es una de las funciones de activación más comúnmente utilizadas en las redes neuronales, especialmente en las redes neuronales convolucionales (CNNs). Su popularidad se debe a su simplicidad y eficacia en la introducción de no linealidades en el modelo, lo que ayuda a aprender funciones más complejas y facilita el proceso de aprendizaje.

![ReLU](https://jacar.es/wp-content/uploads/2023/03/FuncionRELU.png)

Funcionamiento de la ReLU
La función ReLU se define matemáticamente como:

$$ ReLU(x)=max(0,X) $$

Esto significa que la función ReLU toma un valor de entrada  y devuelve cero si x es menor que cero, y devuelve x si x es mayor o igual a cero. Visualmente, la función es una línea que pasa por el origen con una pendiente de 1 para valores positivos y una pendiente de 0 para valores negativos.

### Propiedades y Ventajas de la ReLU
1. No Saturación: A diferencia de las funciones de activación tradicionales como la sigmoide o la tanh, la ReLU no sufre de problemas de saturación (donde las derivadas tienden a cero) para valores positivos. Esto significa que durante el entrenamiento, los gradientes no desaparecen tan rápidamente, lo que es una ventaja para el entrenamiento de redes profundas.

2. Computacionalmente Eficiente: La ReLU es muy simple de calcular, ya que solo requiere una comparación y posiblemente un cambio a cero. No hay funciones exponenciales, como en la sigmoide o tanh, que son computacionalmente más costosas.

3. Esparsidad: Debido a que la ReLU anula todos los valores negativos, se activan menos neuronas en un momento dado. Esta esparsidad puede hacer que la red sea más eficiente y pueda llevar a una mejor representación de características y a un mejor rendimiento general.

### Desventajas de la ReLU
1. Neuronas Muertas: Si un valor de entrada es negativo, la ReLU lo convierte en cero, y durante el proceso de backpropagation, ninguna actualización de peso ocurre en esas neuronas. Esto puede resultar en "neuronas muertas" que nunca se activan durante el entrenamiento, lo que puede reducir la capacidad de aprendizaje de la red.

2. Soluciones: Para mitigar el problema de las neuronas muertas, se han propuesto variantes de la ReLU, como la Leaky ReLU o la Parametric ReLU (PReLU), que permiten un pequeño gradiente cuando el valor de entrada es negativo, evitando así que las neuronas mueran completamente.

### Uso en CNNs
En una CNN, las capas de activación ReLU generalmente siguen inmediatamente a las capas convolutivas. Esto ayuda a introducir no linealidades en el modelo después de que las características lineales han sido extraídas por las convoluciones, permitiendo que la red aprenda soluciones más complejas y efectivas para tareas de clasificación y reconocimiento.

La combinación de capas convolutivas y ReLU ha demostrado ser muy efectiva para el análisis de imágenes y otros tipos de datos que se benefician de la percepción jerárquica de características.

## Capa de Pooling
El Max Pooling es una técnica crucial dentro de las redes neuronales convolucionales (CNNs) que se utiliza generalmente después de las capas convolutivas y de activación (como ReLU). La principal función del max pooling es reducir la dimensionalidad espacial de los mapas de características, lo que ayuda a disminuir la cantidad de parámetros y la computación necesaria en la red, a la vez que introduce cierta invarianza a la traslación en las características detectadas.

### ¿Cómo Funciona el Max Pooling?
1. Operación de Agrupación: Selecciona una ventana o un "kernel" de un tamaño específico, por ejemplo, 2x2 o 3x3, que se desliza sobre el mapa de características entrada.

2. Muestreo: En cada posición de la ventana, el max pooling toma el valor máximo de los valores que están dentro de la ventana. Este valor máximo es el que representa esa región en el mapa de características resultante.

3. Desplazamiento: El kernel se mueve a lo largo del mapa de características original, usualmente sin superposición (stride igual al tamaño del kernel), y repite el proceso de tomar el máximo en cada nueva posición.

### Ventajas del Max Pooling
* Reducción de Dimensiones: Al tomar solo el valor máximo dentro de una ventana local, se reduce significativamente la cantidad de datos, lo que ayuda a evitar el sobreajuste y reduce el costo computacional y de memoria.

* Invariancia a Traslaciones Menores: Al usar el máximo de una región, el resultado es menos sensible a traslaciones y distorsiones menores en la imagen de entrada. Esto es crucial para tareas de reconocimiento de imágenes donde la misma característica puede aparecer en diferentes lugares.

* Enfoca Características Importantes: Al conservar solo los valores máximos, se enfatizan las características más prominentes, lo que puede ser más útil para la clasificación posterior.

### Desventajas del Max Pooling
* Pérdida de Información: Como solo el valor máximo se conserva dentro de cada ventana, se pierde información sobre la apariencia exacta y la posición de las características dentro de esa ventana.

* Falta de Sensibilidad a la Posición Exacta: Aunque la invariancia a la traslación puede ser ventajosa, también significa que la red puede perder sensibilidad a la posición exacta de las características dentro de la imagen, lo cual podría ser importante en algunos contextos.

### Variantes
Existen alternativas al max pooling que intentan mitigar algunas de sus desventajas. Por ejemplo:

* Average Pooling: En lugar de tomar el máximo, calcula el promedio de los valores en la ventana. Esto conserva más información del entorno, pero puede no ser tan efectivo para destacar características fuertes como el max pooling.

* Global Pooling: Calcula un resumen (como el máximo o el promedio) de toda la característica en lugar de regiones locales, útil en algunas arquitecturas modernas de CNN para reducir las dimensiones antes de la clasificación final.

El max pooling sigue siendo una técnica muy popular y efectiva en muchas arquitecturas de CNN, especialmente en aquellas dedicadas a tareas de visión por computadora, debido a su simplicidad y a los beneficios que ofrece en términos de reducción de la complejidad y mejora de la robustez del modelo.

## Flattening
El proceso de flattening en las redes neuronales convolucionales (CNNs) es un paso crucial que se realiza antes de pasar los datos a través de las capas completamente conectadas (también conocidas como capas densas). Después de que la imagen ha sido procesada por varias capas convolutivas y de pooling, que extraen y resumen las características relevantes de la imagen, el mapa de características resultante necesita ser transformado en un formato adecuado para realizar la clasificación final. Aquí es donde entra en juego el flattening.

### ¿Cómo Funciona el Flattening?
El flattening es un proceso relativamente simple pero vital:

* Transformación: Convierte el mapa de características 2D o 3D (en caso de múltiples canales o filtros) en un vector 1D. Es decir, toma todas las filas y columnas de cada mapa de características y las alinea en un único vector largo.

* Preparación para Capas Densas: Este vector se convierte entonces en la entrada para las capas densas. Las capas densas funcionan con vectores de entrada y no con matrices de entrada, de modo que el flattening es esencial para adaptar los datos extraídos de las capas convolucionales y de pooling a las necesidades estructurales de las capas densas.

### Ejemplo de Flattening

Imagina que tenemos un mapa de características de dimensiones 10x10x32 después de varias capas convolutivas y de pooling. Esto significa que tenemos 32 mapas de características separados, cada uno de 10x10. Al aplicar flattening, reorganizamos estos datos en un solo vector de 10×10×32=3200 10×10×32=3200 elementos.

### Ventajas del Flattening
* Interconexión Completa: Permite que las características aprendidas por las convoluciones y agrupaciones sean utilizadas para la clasificación o regresión en capas densas, donde cada característica puede interactuar con las demás a través de conexiones completas.

* Flexibilidad: Facilita la transición entre el procesamiento basado en características locales (capas convolutivas y de pooling) y el procesamiento global (capas densas), lo que es crucial para tareas como la clasificación de imágenes.

### Desventajas del Flattening
* Pérdida de Información Espacial: Al reorganizar las características espaciales en un vector, se pierde la información sobre cómo estaban organizadas espacialmente estas características en la imagen original.

* Alto Número de Parámetros: Al convertir mapas de características en vectores largos y luego alimentarlos a capas densas, se puede aumentar significativamente el número de parámetros en la red, lo que puede llevar a un mayor riesgo de sobreajuste y aumentar los requerimientos computacionales.

### Uso Práctico
En las arquitecturas modernas de CNN, a veces se evita el flattening mediante el uso de capas de pooling global antes de las capas densas, que reducen cada mapa de características a un solo valor, manteniendo la estructura general pero reduciendo enormemente la dimensión de la entrada a las capas densas.

El flattening es una técnica estándar que prepara el terreno para las decisiones finales en una CNN, facilitando el análisis complejo y la clasificación basada en las características ricas y jerárquicas aprendidas por las capas anteriores de la red.

## Full Connection
Las capas completamente conectadas, también conocidas como capas densas o "fully connected layers" en inglés, son un componente esencial en las redes neuronales, incluyendo las redes neuronales convolucionales (CNNs). Estas capas juegan un papel crucial después de que las operaciones de convolución y pooling han extraído y condensado las características importantes de los datos de entrada, como imágenes o series de tiempo.

### Funcionamiento de las Capas Completamente Conectadas
En una capa completamente conectada, cada neurona está conectada a todas las neuronas en la capa anterior. Esto significa que las entradas de estas capas son combinaciones lineales de las salidas de la capa anterior, seguidas generalmente por una función de activación no lineal como ReLU, sigmoid o tanh.

### Estructura de una Capa Densa
Vector de Entrada: Generalmente, la entrada a una capa completamente conectada proviene de una capa de "flattening" que convierte los mapas de características multidimensionales en un único vector largo, o de una capa de pooling que reduce las dimensiones de los datos.

* Matriz de Pesos: Cada conexión entre neuronas de capas consecutivas tiene un peso asociado. La capa densa aprende estos pesos durante el entrenamiento de la red.

* Bias: Cada neurona en una capa completamente conectada puede tener un término de bias adicional, que se añade a la suma ponderada de las entradas.

* Función de Activación: Después de calcular la suma ponderada de las entradas y los biases, se aplica una función de activación. Esta función introduce la no linealidad necesaria para que la red pueda aprender funciones complejas.

### Proceso de una Capa Densa
Se calcula la suma ponderada de las entradas para cada neurona: 
$$ z = w_1x_1 + w_2x_2 + ... + w_nx_n + b $$

 ### Importancia en las CNNs
En el contexto de las CNNs, las capas completamente conectadas suelen ubicarse cerca del final de la red. Después de que las capas convolutivas y de pooling han detectado características y reducido la dimensión de los datos, las capas densas interpretan estas características para realizar tareas como clasificación o regresión. Aquí es donde la red toma decisiones basadas en las características extraídas.

### Ventajas de las Capas Densas
Interpretación de Características: Pueden aprender combinaciones complejas de características, lo que es crucial para la toma de decisiones en tareas de clasificación.

* Flexibilidad: Pueden ser utilizadas para convertir cualquier salida de tamaño fijo en el número requerido de salidas para una tarea específica, como clasificar las entradas en diferentes categorías.
### Desafíos
* Número de Parámetros: Pueden tener un gran número de parámetros, especialmente si la entrada es grande, lo que puede llevar a sobreajuste y a un incremento en los requisitos computacionales.

* Necesidad de Regularización: A menudo requieren técnicas de regularización como dropout o L2 * regularization para prevenir el sobreajuste.

Las capas completamente conectadas son fundamentales en la arquitectura de muchas redes neuronales, proporcionando la base para la toma de decisiones y la interpretación de datos a partir de las características aprendidas por la red.

## Softmax
Softmax es una función de activación que se aplica generalmente en la capa de salida de las redes neuronales cuando se enfrentan a problemas de clasificación multiclase. Su objetivo principal es transformar los valores de entrada (logits) en valores de salida que sumen un total de 1, de modo que estos últimos puedan ser interpretados como probabilidades.

### Funcionamiento de la Función Softmax
La función Softmax se define matemáticamente como:
$$ \sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} $$
Donde:
* $z$ es el vector de logits (los valores de entrada a la función Softmax).
* $Ki$ es el logit para la clase $i$.
* $e^zk$ es la clase específica para la que se está calculando la probabilidad.

### Proceso de Cálculo en Softmax
1. Exponenciación: Se calcula el exponencial de cada logit en el vector de entrada.
2. Normalización: Se divide cada exponencial por la suma de todos los exponenciales en el vector, lo que garantiza que la suma total de las probabilidades sea 1.

### Ejemplo de Uso de Softmax
Imagina que tienes una red neuronal que intenta clasificar imágenes en tres categorías. Después de todas las transformaciones en la red, llegas a tres logits: 
[2.0,1.0,0.1]. Al aplicar Softmax, cada logit se exponencia y luego se normaliza dividiendo por la suma de las exponenciales:

* $e^2.0, e^1.0, e^0.1 = [7.4, 2.7, 1.1]$
* Suma estos valores.
* Divide cada exponencial por esta suma para obtener las probabilidades.

### Ventajas de Softmax
* Interpretación Probabilística: Convierte logits que pueden ser cualquier número real, positivo o negativo, en valores que son probabilidades interpretables.

* Diferenciación Clara: Debido a la exponenciación, las diferencias entre los logits se amplifican, lo que puede hacer más clara la elección de la clase más probable.
### Aplicaciones Comunes
* Clasificación Multiclase: Softmax es estándar en la capa final de clasificadores donde se necesita seleccionar entre múltiples categorías.

* Modelos de Lenguaje: Utilizada en modelos que generan texto para seleccionar la próxima palabra más probable.

* Redes Neuronales Recurrentes: En tareas como la traducción automática, donde la salida de cada paso de tiempo puede ser una de varias palabras o caracteres.

La función Softmax es una herramienta poderosa para finalizar las redes de clasificación, proporcionando un mecanismo claro y matemáticamente fundado para tomar decisiones de clasificación basadas en las características aprendidas por la red.

## Entropia cruzada

La entropía cruzada es una función de pérdida ampliamente utilizada en el aprendizaje automático, especialmente en problemas de clasificación. Es fundamental cuando utilizamos la función de activación Softmax en la capa de salida de una red neuronal, ya que mide la diferencia entre las distribuciones de probabilidad real y la predicha por el modelo. Es decir, evalúa qué tan bien las probabilidades predichas por la red coinciden con las etiquetas reales en formato de probabilidad.

### Definición de Entropía Cruzada
La entropía cruzada entre dos distribuciones de probabilidad, una verdadera 
$p$ y una predicha $q$ , se define para la clasificación como:
$$ H(p,q) = -\sum_{i} p(x) \log(q(x)) $$
Donde:
* $p(x)$ es la probabilidad real de la clase $x$.
* $q(x)$ es la probabilidad predicha de la clase $x$.

### ¿Cómo Funciona la Entropía Cruzada?
En el contexto de una red neuronal para clasificación:
 * Propiedades Verdaderas: $p(x)$ es un vector de probabilidad "one-hot" que contiene un 1 en la posición de la clase verdadera y 0 en todas las demás posiciones.
* Propiedades Predichas: $q(x)$ es el vector de probabilidades predichas por la red, generalmente después de aplicar Softmax en la capa de salida.
* Cálculo de la Pérdida: La entropía cruzada mide la diferencia entre las dos distribuciones de probabilidad, penalizando las predicciones incorrectas con una pérdida más alta.

### Ejemplo
Supongamos una clasificación con tres clases. Si la clase verdadera es la primera, 
$p$ será [1, 0, 0] y si el modelo predice las probabilidades como [0.8, 0.1, 0.1], la entropía cruzada se calculará como:
$$ H(p,q) = -(1  \log(0.8) + 0 \log(0.1) + 0 \log(0.1) = -\log(0.8) = 0.223) $$

### Ventajas de la Entropía Cruzada
* Sensibilidad a la Probabilidad: Al penalizar las diferencias entre las probabilidades verdaderas y las predichas, promueve un modelo que no solo acierta la clasificación sino que también es seguro de sus predicciones.

* Gradientes Efectivos: En la práctica, proporciona gradientes claros y efectivos durante el entrenamiento, evitando problemas como el de los gradientes que desaparecen, comunes con otras funciones de pérdida en el contexto de probabilidades.

### Aplicaciones
* Modelos de Clasificación: Utilizada en casi todos los modelos de clasificación donde las salidas pueden interpretarse como probabilidades.

* Modelos de Lenguaje y Generativos: En estos modelos, la entropía cruzada ayuda a comparar la distribución de palabras o características generadas contra las distribuciones verdaderas.

La entropía cruzada es una herramienta esencial en la caja de herramientas de machine learning, permitiendo ajustar modelos de manera que sus salidas reflejen no solo decisiones correctas sino también certezas probabilísticas adecuadas sobre esas decisiones.






