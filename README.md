# Water Quality Prediction
- Water quality prediction using machine learning algorithms and neural networks (MLP and KNN)
- **Binary classification problem**

  - **1 (True) - The water is potable**
  - **0 (False) - The water is not potable**

- **Project workflow**

<img src="https://user-images.githubusercontent.com/61973790/224288065-a69dadb5-69bd-4df6-84e4-8c3c9cc18a19.png" width="90%" height="90%">

## Used tools and technologies
- Python 3
- Spyder inside the Anaconda environment
- Python libraries:

  - **pandas** - loading a dataset (.csv)
  - **numpy** - data processing
  - **matplotlib** - graphical representation of the results
  - **sklearn** - machine learning library, contains predefined neural network models and machine learning algorithms (MLP and KNN)
  
## Datasets

### Dataset 1
- **3275** data records
- 12 water characteristics - pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability, Carcinogenics, medical_waste
- Classification variable - **Potability**
- **Training set: 70%**
- **Validation set: 30%**
- **The columns of the last two water characteristics (Carcinogenics, medical_waste) were removed** because they did not contain any values (they have no influence on model learning)
- **Example 1** - Replace all Nan values with 0 - df1
- **Example 2** - Remove all rows with at least 1 Nan value - df2

### Dataset 2
- **7995** data records
- 21 water characteristics - aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, flouride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, uranium and is_safe
- Classification variable - **is_safe**
- **Training set: 70%**
- **Validation set: 30%**

## Models

### MultiLayer Perceptron (MLP)
- Neural network where the mapping between inputs and output is non-linear
- It consists of input and output layers, and one or more hidden layers with many neurons stacked together
- Each node, except the input nodes, has non-linear activation function
- Uses backpropagation as supervisory learning technique

<img src="https://user-images.githubusercontent.com/61973790/225048969-8520aa58-3db6-4e7a-86ab-1359f08cc0ba.jpg" width="40%" height="40%">

### K-Nearest Neighbors (KNN) 
- Classification algorithm and one of the most used machine learning method technique
- Non-parametric algorithm that classifies an input into the one of classification groups using the Euclidean distances to k-nearest neighbors

<img src="https://user-images.githubusercontent.com/61973790/225053505-b9a0d24d-f3d0-4472-8274-146f015b63c9.png" width="75%" height="75%">

## Training
- Models training (MLP, KNN) was done using different parameters and datasets
- The training was completed after approximately 100 epochs because the loss function has not shown significant changes (for all tested combinations of parameters)

## Validation results

### Dataset 1
- Satisfactory results **were not obtained**
- **MLP model** - always predicts only one output value ( 0 or 1) regardless of the parameters used
- **KNN model** - better results (K=1), but still a high percentage of error


### Dataset 2
- Satisfactory results **were obtained**
- **MLP model**
  - **"Water is not potable" - 98% correctly classified data**
  - **"Water is potable" - 71% correctly classified data**
  
 <table>
  <tr>
    <td align="center">
      <table>
        <tr>
           <td colspan="2" align="center"><b>MLP</b></td>
        </tr>
        <tr>
           <td align="center"><b>Parameter</b></td>
           <td align="center"><b>Value</b></td>
        </tr>
        <tr>
          <td align="center">hidden_layer_sizes</td>
          <td align="center">(500, 300)</td>
        </tr>
        <tr>
          <td align="center">alpha</td>
          <td align="center">0.0001</td>
        </tr>
        <tr>
          <td align="center">solver</td>
          <td align="center">adam</td>
        </tr>
        <tr>
          <td align="center">activation</td>
          <td align="center">relu</td>
        </tr>
        <tr>
          <td align="center">max_iter</td>
          <td align="center">1000</td>
        </tr>
        <tr>
          <td align="center">verbose</td>
          <td align="center">True</td>
        </tr>
      </table>
    </td>
    <td align="center">
     <img src="https://user-images.githubusercontent.com/61973790/224268265-41b4b19d-5179-4186-aa4a-14c2b8926726.png" width="70%" height="70%">
    </td>
  </tr>
</table>

- **KNN model (K=5)**
  - **"Water is not potable" - 98% correctly classified data**
  - **"Water is potable" - 34% correctly classified data**
  
  <img src="https://user-images.githubusercontent.com/61973790/224275226-c99039d5-a5f0-49aa-b5df-cbd90e25217f.png" width="45%" height="45%">
  
## Conclusions
- MLP showed better prediction results than KNN because the MLP is a much more complex algorithm that takes into account many more parameters and uses an advanced methods such as backpropagation
- KNN has a higher execution speed compared to MLP, especially if the MLP uses a "slower" activation function (eg logistic) and many hidden layers with a large number of neurons
- Dataset 2 contained more data with a higher correlation rate compared to the Dataset 1 (better prediction results using data from the Dataset 2)
- Higher number of hidden layers and neurons doesn't mean better results
- The determination of model parameters depends on the dataset used and the relationship between these data (eg is the data correlated)
