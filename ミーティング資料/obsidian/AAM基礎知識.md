# L7 Assignment

  

## Step 1: Find the 2 Nearest Neighbors

  

Ada is Female with Total Cholesterol = 172.

  

| Name  | Cholesterol | BMI  | Difference |

|-------|------------:|-----:|-----------:|

| Daisy | 166         | 22.7 | 6          |

| Kate  | 181         | 26.3 | 9          |

| Lily  | 201         | 28.0 | 29         |

| Mary  | 215         | 28.6 | 43         |

  

Therefore, the two nearest neighbors are **Daisy** and **Kate**.

  

## Step 2: Calculate the Weights

  

The weights are proportional to the reciprocal of the cholesterol difference:

  

- Daisy: $\frac{1}{6}$

- Kate: $\frac{1}{9}$

  

Normalized weights:

  

$$\text{Daisy} = \frac{1/6}{1/6 + 1/9} = 0.6$$

  

$$\text{Kate} = \frac{1/9}{1/6 + 1/9} = 0.4$$

  

## Step 3: Estimate Ada's BMI

  

$$\text{BMI}(\text{Ada}) = 0.6 \times 22.7 + 0.4 \times 26.3 = 13.62 + 10.52 = 24.14$$

  

## Final Answer

  

Ada's estimated BMI is **24.14**.