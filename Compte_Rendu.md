# 📊 Projet : Algorithmes de Prédiction Supervisée avec Scikit‑Learn

## Auteur

Projet réalisé dans le cadre d'un TP d'introduction au **Machine Learning**.

---

# 1. Introduction

L'objectif de ce projet est d'explorer plusieurs **algorithmes d'apprentissage supervisé** en utilisant la bibliothèque Python **scikit‑learn**.

Nous avons implémenté différents modèles de **classification** et de **régression** afin de :

* comprendre leur fonctionnement
* comparer leurs performances
* observer les différences entre modèles linéaires et non linéaires

Les expériences ont été réalisées dans **Google Colab** et le code est organisé sous forme de **notebook**.

---

# 2. Environnement et bibliothèques utilisées

Les principales bibliothèques utilisées sont :

* **NumPy** : calcul scientifique
* **Pandas** : manipulation de données
* **Matplotlib** : visualisation
* **Scikit‑learn** : algorithmes de Machine Learning

Installation des dépendances :

```python
!pip install scikit-learn numpy pandas matplotlib scipy
```

Import des bibliothèques :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
```

---

# 3. Chargement des jeux de données

Plusieurs datasets standards fournis par **scikit‑learn** ont été utilisés.

```python
iris = datasets.load_iris()
wine = datasets.load_wine()
breast = datasets.load_breast_cancer()
```

### Description des datasets

| Dataset       | Type           | Taille           | Objectif                          |
| ------------- | -------------- | ---------------- | --------------------------------- |
| Iris          | Classification | 150 observations | classification de fleurs          |
| Wine          | Classification | 178 observations | classification de vins            |
| Breast Cancer | Classification | 569 observations | diagnostic médical                |
| Diabetes      | Régression     | 442 observations | prédiction d'une valeur numérique |

---

# 4. Préparation des données

Les données sont séparées en deux ensembles :

* **Training set (80%)** : utilisé pour entraîner le modèle
* **Test set (20%)** : utilisé pour évaluer la performance

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Pour certains modèles, les variables doivent être **normalisées** avec `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

---

# 5. Algorithmes de Classification

## 5.1 Régression Logistique

La régression logistique est un modèle linéaire utilisé pour les problèmes de **classification**.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Résultat

Accuracy approximative : **≈ 96‑97 %**

---

## 5.2 Decision Tree

Les **arbres de décision** divisent les données en plusieurs branches à l'aide de règles logiques.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
```

### Avantages

* très interprétable
* visualisable

---

## 5.3 Random Forest

Random Forest est un **ensemble d'arbres de décision** entraînés sur des sous‑échantillons des données.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### Résultat

Accuracy typique : **≈ 97 %**

---

## 5.4 Gradient Boosting

Gradient Boosting construit plusieurs arbres **séquentiellement** afin de corriger les erreurs du modèle précédent.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

Ce modèle est souvent très performant.

---

## 5.5 Support Vector Machine (SVM)

Les **SVM** cherchent un hyperplan séparant les classes avec une marge maximale.

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf")
model.fit(X_train, y_train)
```

---

## 5.6 K‑Nearest Neighbors (KNN)

KNN classe un point en fonction de ses **k voisins les plus proches**.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

---

## 5.7 Naive Bayes

Naive Bayes est un classifieur probabiliste basé sur le **théorème de Bayes**.

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

---

# 6. Algorithmes de Régression

Les modèles de régression permettent de prédire une **valeur numérique continue**.

Dataset utilisé : **Diabetes dataset**.

---

## 6.1 Régression Linéaire

La régression linéaire modélise une relation linéaire entre les variables.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

Métrique utilisée : **R² score**.

---

## 6.2 Ridge Regression

Ridge ajoute une **pénalisation L2** pour éviter le sur‑apprentissage.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
```

---

## 6.3 Lasso Regression

Lasso utilise une **pénalisation L1** qui peut annuler certains coefficients.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
```

---

## 6.4 Elastic Net

Elastic Net combine **L1 + L2**.

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

---

## 6.5 Support Vector Regression (SVR)

Extension du SVM pour les problèmes de régression.

```python
from sklearn.svm import SVR

model = SVR(kernel="rbf")
```

---

## 6.6 MLP Regressor

MLP Regressor est un **réseau de neurones artificiel** pour la régression.

```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(128,64,32))
```

---

# 7. Régression Polynomiale

La régression polynomiale permet de modéliser des relations **non linéaires**.

```python
from sklearn.preprocessing import PolynomialFeatures
```

Nous avons testé plusieurs degrés :

| Degré | Interprétation                |
| ----- | ----------------------------- |
| 1     | modèle linéaire               |
| 2     | relation non linéaire modérée |
| 3     | risque d'overfitting          |

Exemple de pipeline :

```python
Pipeline([
 ("poly", PolynomialFeatures(degree=2)),
 ("sc", StandardScaler()),
 ("ridge", Ridge())
])
```

---

# 8. Comparaison des modèles

### Classification

| Modèle              | Performance approximative |
| ------------------- | ------------------------- |
| Logistic Regression | ~96‑97%                   |
| Decision Tree       | ~93‑96%                   |
| Random Forest       | ~97%                      |
| Gradient Boosting   | ~97‑98%                   |
| SVM                 | ~97%                      |
| KNN                 | ~96‑97%                   |
| Naive Bayes         | ~94‑96%                   |

---

### Régression

| Modèle            | Score R² approximatif |
| ----------------- | --------------------- |
| Linear Regression | ~0.48                 |
| Ridge             | ~0.49                 |
| Lasso             | ~0.45                 |
| Elastic Net       | ~0.47                 |
| SVR               | ~0.50                 |
| MLP Regressor     | ~0.52                 |

---

# 9. Conclusion

Ce projet nous a permis de :

* découvrir plusieurs **algorithmes majeurs du Machine Learning**
* comprendre la différence entre **classification et régression**
* expérimenter les modèles avec **scikit‑learn**

Les résultats montrent que :

* les méthodes **ensemble** (Random Forest, Gradient Boosting) donnent souvent les meilleures performances
* la régularisation (**Ridge / Lasso**) améliore la généralisation
* les réseaux de neurones (**MLP**) peuvent capturer des relations complexes

Ce travail constitue une introduction pratique aux méthodes de **prédiction supervisée** utilisées en data science.

---

# 10. Références

* Documentation officielle **Scikit‑Learn**
* Datasets fournis par **sklearn.datasets**
* Notebook Google Colab du projet
