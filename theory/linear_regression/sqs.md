### 1. Función de Costo

Para un modelo de regresión lineal, la función de costo $J(\theta)$ se define generalmente como la suma de los errores al cuadrado de cada predicción con respecto al valor real. La función de costo se expresa como:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$

Donde:

- $J(\theta)$ es la función de costo que queremos minimizar.
- $h_\theta(x)$ es la hipótesis del modelo, definida como $h_\theta(x) = \theta^T x$ para la regresión lineal.
- $n$ es el número de ejemplos en el conjunto de entrenamiento.
- $x^{(i)}, y^{(i)}$ son los ejemplos de entrenamiento individuales.
- $\theta$ son los parámetros del modelo que estamos aprendiendo.

### 2. Descenso de Gradiente

El objetivo del descenso de gradiente es encontrar el valor óptimo de $\theta$ que minimiza la función de costo $J(\theta)$. Para hacer esto, actualizamos cada parámetro $\theta_j$ utilizando la siguiente regla:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

La derivada parcial de $J(\theta)$ con respecto a $\theta_j$ para la función de costo de mínimos cuadrados es:

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$
Recordar que la derivada nos da el punto tangente a la funcion y siempre en la direccion creciente, por tanto buscamos la direccion opuesta porque queremos minimizar $J(\theta)$

### 3. Criterios de Detención

El proceso de actualización continúa hasta que se cumpla una de las siguientes condiciones:

1. **Convergencia:** Si $J(\theta)$ disminuye muy poco (menos que un umbral predeterminado) en una iteración, se considera que la función ha convergido y se detiene el algoritmo.
2. **Número máximo de iteraciones:** Se puede especificar un número máximo de iteraciones como criterio de detención.
3. **Cambio en $\theta$:** Si los valores de $\theta$ cambian muy poco entre iteraciones, esto también podría considerarse un criterio de detención.

### 4. Elección de la Tasa de Aprendizaje $\alpha$

La elección de $\alpha$ es crítica para la eficiencia del algoritmo. Un $\alpha$ demasiado pequeño hará que la convergencia sea lenta, mientras que un $\alpha$ demasiado grande podría causar oscilaciones o divergencia.