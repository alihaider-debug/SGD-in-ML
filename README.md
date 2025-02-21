Name: Ali Haider

Roll No: 21I-1522

# SGD-in-ML

# Stochastic Gradient Descent (SGD) for Linear Regression  
This repository contains Python implementations of **Stochastic, Batch, and Mini-Batch Gradient Descent** for **Linear Regression**. It also includes experiments on the **effect of learning rate and momentum on convergence**.

## 📂 Contents
- **SGD for Linear Regression:** Implementing SGD, Batch, and Mini-Batch Gradient Descent.
- **Effect of Learning Rate:** Observing how different learning rates impact model convergence.
- **Effect of Momentum:** Understanding how momentum influences optimization speed.
- **Final Comparisons:** Comparing convergence speed and final model parameters.

## 📥 How to Run  
This project is designed for **Google Colab**. Open the provided `.ipynb` file in Colab, execute all cells, and observe the results.  
Ensure that datasets are downloaded within the Colab notebook.

## 🔗 Google Colab Notebook
Click below to open and execute the notebook in Google Colab:  
[![Open in Colab]([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alihaider-debug/SGD-in-ML/blob/main/21I-1522_A02.ipynb))]

---

## 📌 Step 1: Deriving the Gradient Update Rule for Simple Linear Regression Using MSE Loss  

### 1️⃣ Defining the Linear Regression Model  
A simple linear regression model is given by:  

\[
\hat{y} = w x + b
\]

where:  
- \(\hat{y}\) is the predicted value,  
- \(x\) is the input feature,  
- \(w\) is the weight (coefficient),  
- \(b\) is the bias (intercept).  

### 2️⃣ Defining the Loss Function  
We use the **Mean Squared Error (MSE)** as the loss function:  

\[
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - (\hat{w} x_i + \hat{b}))^2
\]

where:  
- \(N\) is the number of training samples,  
- \(y_i\) is the actual output for the \(i\)th sample,  
- \(\hat{y_i} = w x_i + b\) is the predicted output.  

### 3️⃣ Compute Partial Derivatives (Gradients)  
To minimize the loss, we compute the partial derivatives of \(L\) with respect to \(w\) and \(b\).  

Gradient with respect to \(w\):  

\[
\frac{\partial L}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} -2 x_i (y_i - (\hat{w} x_i + \hat{b}))
\]

Gradient with respect to \(b\):  

\[
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} -2 (y_i - (\hat{w} x_i + \hat{b}))
\]

### 4️⃣ Gradient Descent Update Rule  
Using **Gradient Descent**, we update \(w\) and \(b\) as follows:  

\[
w := w - \alpha \frac{\partial L}{\partial w}
\]

\[
b := b - \alpha \frac{\partial L}{\partial b}
\]

where:  
- \(\alpha\) is the **learning rate**, controlling step size in the optimization process.  

---

## 📊 Comparisons and Observations  
### 🟢 Effect of Learning Rate  
- Higher learning rates converge **faster** but risk **overshooting** the optimal solution.  
- Lower learning rates converge **slowly** but may provide **better stability**.  
- A **moderate learning rate** balances speed and accuracy.

### 🔵 Effect of Momentum  
- Adding **momentum** helps **reduce oscillations** and speeds up convergence.  
- A **high momentum** may overshoot, while a **low momentum** may slow down learning.

---

## 🚀 Conclusion  
This project demonstrates how **SGD, Batch, and Mini-Batch Gradient Descent** impact **convergence speed** and **final model parameters**.  
By adjusting **learning rates and momentum**, we can fine-tune optimization performance.

---

## 🛠️ Technologies Used  
- **Python**  
- **Google Colab**  
- **NumPy, Matplotlib, Scikit-Learn**  


## 🔗 GitHub Repository  
## 🔗 GitHub Repository  
(## 🔗 GitHub Repository  
[GitHub Repository Link](https://github.com/alihaider-debug/SGD-in-ML))
