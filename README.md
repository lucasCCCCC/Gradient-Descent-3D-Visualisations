# Machine Learning Visualisers

Linear Regression & Logistic regression, two commonly used supervised machine leanrning algorithms will be implemented and visualised

For each algorithm the following will be shown:
- A gradient descent animation, where the loss function is finding the optimal solution (local minima)
- A animation showing the model learning what the best fit of the data-set is

## Linear Regression 

Model used: ![image](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Clarge%20y%20%3D%20w_1%20x%20&plus;%20w_0)


Loss function used - L2 Loss: 

![image](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Clarge%20g%28w_0%2C%20w_1%29%20%3D%20%5Cdisplaystyle%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%7D%20%28%28w_1x%5E%7B%5Bn%5D%7D&plus;w_0%29%20-%20y%5E%7B%5Bn%5D%7D%29)

During gradient descent, we take partial derivatives of our loss function with respect to each weight

![image](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Clarge%20%5Cfrac%7B%5Cpartial%20g%28w_0%2C%20w_1%29%7D%7B%5Cpartial%20w_0%7D%20%3D%20%5Cfrac%7B2%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28w_1x%5E%7B%5Bn%5D%7D&plus;w_0%29-y%5E%7B%5Bn%5D%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20g%28w_0%2C%20w_1%29%7D%7B%5Cpartial%20w_1%7D%20%3D%20%5Cfrac%7B2%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28%28w_1x%5E%7B%5Bn%5D%7D&plus;w_0%29-y%5E%7B%5Bn%5D%7D%29x%5E%7B%5Bn%5D%7D%29)

We then use these values to continuously update our weights until convergence

![image](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Clarge%20w_0%20%3D%20w_0%20-%20%5Calpha%20%5Ccdot%20%28%28w_1x%5E%7B%5Bn%5D%7D&plus;w_0%29-y%5E%7B%5Bn%5D%7D%29%20%5C%5C%20w_1%20%3D%20w_1%20-%20%5Calpha%20%5Ccdot%20%28%28w_1x%5E%7B%5Bn%5D%7D&plus;w_0%29-y%5E%7B%5Bn%5D%7D%29x%5E%7B%5Bn%5D%7D)


### Example visualisation over a given dataset:


<table>
  <tr>
    <td>On the 60<sup>th</sup> iteration<img src="images/img.png" width="500"></td>
    <td>On the 270<sup>th</sup> iteration<img src="images/img_1.png" width="500"></td>
  </tr>
    <tr>
    <td>On the 510<sup>th</sup> iteration<img src="images/img_2.png" width="500"></td>
    <td>On the 990<sup>th</sup> (final) iteration<img src="images/img_3.png" width="500"></td>
  </tr>
 </table>
 
 
## Logistic Regression 

Model used - Sigmoid function: 

![image](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Clarge%20y%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%28w_1x&plus;w_0%29%7D%7D)

Loss function used - Cross entropy loss function:

![image](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Clarge%20%5Ctext%7Bwhere%20%7D%20%5Csigma%28z%29%20%5Ctext%7B%20is%20our%20sigmoid%20function%20and%20%7D%20z%20%3D%20w_1x%5E%7B%5Bn%5D%7D&plus;w_0%20%5C%5C%20g%20%3D-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%7B%28y%5E%7B%5Bn%5D%7D%5Clog%28%5Csigma%28z%29%29%20&plus;%20%281%20-%20y%5E%7B%5Bn%5D%7D%29%5Clog%281%20-%20%5Csigma%28z%29%29%29%7D)

Each of our weights are updated until convergence:

![image](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Clarge%20w_0%20%3D%20w_0%20-%20%5Calpha%20%5Ccdot%20%28%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28%5Csigma%28z%29%20-%20y%5E%7B%5Bn%5D%7D%29%29%5C%5C%20w_1%20%3D%20w_1%20-%20%5Calpha%20%5Ccdot%20%28%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28%5Csigma%28z%29%20-%20y%5E%7B%5Bn%5D%7D%29x%5E%7B%5Bn%5D%7D%29%5C%5C)


### Example visualisation over a given dataset:


<table>
  <tr>
    <td>On the 400<sup>th</sup> iteration<img src="images/img_21.png" width="500"></td>
    <td>On the 800<sup>th</sup> iteration<img src="images/img_22.png" width="500"></td>
  </tr>
    <tr>
    <td>On the 1400<sup>th</sup> iteration<img src="images/img_23.png" width="500"></td>
    <td>On the 3000<sup>th</sup> iteration<img src="images/img_24.png" width="500"></td>
  </tr>
 </table>
 
## Imported Libraries:

- matplotlib
- numpy
- celluloid
- scipy
- random

