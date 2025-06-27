# 📊 Bias-Aware User-Based Collaborative Filtering

![C++](https://img.shields.io/badge/language-C%2B%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/license-MIT-green.svg) ![RMS Error](https://img.shields.io/badge/RMS_0.9111-red.svg) ![Runtime](https://img.shields.io/badge/Time_0.2s-orange.svg)

An optimized **bias-aware user-based collaborative filtering** recommendation engine implemented in C++. It computes personalized predictions by combining baseline bias models with neighborhood-based similarity, featuring significance weighting and case amplification.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Algorithm Flow](#-algorithm-flow)
4. [Performance](#-performance)
5. [Prerequisites](#-prerequisites)
6. [Building & Installation](#-building--installation)
7. [Usage](#-usage)
8. [Configuration](#-configuration)
9. [Code Structure](#-code-structure)
10. [License](#-license)
11. [Contributing](#-contributing)
12. [Contact](#-contact)

---

## 🔍 Overview

This project implements a **user-based collaborative filtering** approach that:

* **Models biases** via global mean, user bias, and item bias refined through gradient descent.
* **Computes residuals** and builds a similarity graph using cosine similarity with significance weighting and case amplification.
* **Predicts ratings** by blending baseline estimates with weighted neighbor residuals.

The result is a fast, memory-efficient recommender achieving an RMS error of **0.9111** in **0.2s** on typical datasets.

---

## ✨ Key Features

* **Bias Modeling:** Iterative refinement of user and item biases (8 iterations).
* **Neighborhood Selection:** Top-K (K=190) neighbors per user with significance shrinkage (10) and case amplification (1.3).
* **Efficient Similarity:** Sparse pairwise dot-product computation and max-heap for top-K extraction.
* **Gradient Descent:** Learning rate α=0.01 with regularization λ=0.02 for robust bias estimation.
* **Plain C++:** No external libraries, high-performance code using STL and chrono for timing.

---

## 🏷️ Algorithm Flow

1. **Data Loading**: Read training and test datasets from `stdin`.
2. **Global Mean**: Compute the overall average rating.
3. **Bias Initialization**: Set user and item biases to zero.
4. **Bias Refinement**: Perform `NUM_ITERS` passes of gradient updates:

    * err = rating − (globalMean + userBias\[u] + itemBias\[i])
    * Update biases:

      ```cpp
      userBias[u] += ALPHA * (err - REG * userBias[u]);
      itemBias[i] += ALPHA * (err - REG * itemBias[i]);
      ```
5. **Residual Computation**: For each rating, compute residual = rating − baseline estimate.
6. **Similarity Calculation**:

    * Accumulate dot products for co-rated items.
    * Compute raw cosine similarity.
    * Apply significance weighting: `count / (count + SHRINK)`.
    * Apply case amplification: `|sim|^AMP_FACTOR`.
    * Maintain top-K neighbors via max-heaps.
7. **Prediction**: For each test instance:

    * Compute baseline = globalMean + userBias + itemBias.
    * Aggregate neighbor residuals weighted by similarity.
    * Final rating = baseline + (weightedSum / sumOfWeights).
8. **Output**: Print predictions to `stdout` and log elapsed time to `stderr`.

---

## 🚀 Performance

| Metric      | Value    |
| ----------- | -------- |
| RMS Error   | 0.9111   |
| Runtime     | 0.2 s    |
| Memory      | O(N + M) |
| Neighbors K | 190      |

*Measured on a dataset with \~100k ratings.*

---

## 📋 Prerequisites

* **C++ Compiler**: GCC or Clang with C++11 support
* **Build Tools**: Make (optional)

---

## 🛠️ Building & Installation

```bash
# Clone the repository
git clone https://github.com/<your-user>/BiasAwareUserCF.git
cd BiasAwareUserCF

# Build
g++ -std=c++11 -O3 main.cpp -o bias_cf
```

> Alternatively, use `Makefile`:
>
> ```bash
> make
> ```

---

## 🎮 Usage

Run the executable, providing training and test data via `stdin`:

```bash
./bias_cf < data.txt
```

Where `data.txt` format:

```
train dataset
<userId> <itemId> <rating>
...
test dataset
<userId> <itemId>
...
```

Predicted ratings are printed line-by-line to `stdout`, and execution time is logged to `stderr`.

---

## ⚙️ Configuration

\| Parameter        | Default   | Description |

\|------------------|-----------|-------------------------------------------------|

`|`K`             |`190`    | Number of neighbors per user    |

|`SHRINK`        |`10.0`   | Significance shrinkage factor                   |

|`AMP\_FACTOR`    |`1.3`    | Case amplification exponent                     |

|`NUM\_ITERS`     |`8`      | Iterations of bias refinement                   |

|`ALPHA`(α)      |`0.01`   | Learning rate for gradient updates              |

|`REG`(λ)        |`0.02\`| Regularization coefficient                      |

To adjust parameters, edit the constants at the top of `main.cpp`.

---

## 📂 Code Structure

```
BiasAwareUserCF/
├── main.cpp      # Core implementation
├── Makefile      # Build script (optional)
└── README.md     # This documentation
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit pull requests or open issues for feature requests and bug reports.

---

## 📬 Contact

Maintainer: `<sasan.shafiee.m@gmail.com>`

Enhance your recommender systems with bias-aware filtering! 🚀
