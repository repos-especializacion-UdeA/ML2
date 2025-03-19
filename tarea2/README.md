# Tarea 2


## t-SNE

### Fragmentos de codigo

https://www.kaggle.com/datasets/seshadrikolluri/ansur-ii/data

```py

# Data t-SNE on female ANSUR dataset
df.shape

non_numeric = ['BMI_class', 'Height_class', 'Gender', 'Component', 'Branch']
df_numeric = df.drop(non_numeric, axis=1)
df_numeric.shape


# Fitting t-SNE
from sklearn.manifold import TSNE

m = TSNE(learning_rate=50)
tsne_features = m.fit_transform(df_numeric)
tsne_features[1:4,:]

# Assigning t-SNE features to our dataset
tsne_features[1:4,:]
df['x'] = tsne_features[:,0]
df['y'] = tsne_features[:,1]

# Plotting t-SNE
import seaborn as sns
sns.scatterplot(x="x", y="y", data=df)
plt.show()

# Coloring points according to BMI category
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="x", y="y", hue='BMI_class', data=df)
plt.show()

# Coloring points according to height category
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="x", y="y", hue='Height_class', data=df)
plt.show()
```

### Dimensionality reduction in python

```py
# Creating a feature selector
print(ansur_df.shape)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=1)
sel.fit(ansur_df)
mask = sel.get_support()
print(mask)

# Applying a feature selector
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape)

# Variance selector caveats
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape)

# Normalizing the variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.005)
sel.fit(ansur_df / ansur_df.mean())
mask = sel.get_support()
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape)

# Identifying missing values
pokemon_df.isna()

# Counting missing values
pokemon_df.isna().sum()
pokemon_df.isna().sum() / len(pokemon_df)

# Applying a missing value threshold
# Fewer than 30% missing values = True value
mask = pokemon_df.isna().sum() / len(pokemon_df) < 0.3
print(mask)

# Applying a missing value threshold
reduced_df = pokemon_df.loc[:, mask]
reduced_df.head()

# Pairwise correlation
sns.pairplot(ansur, hue="gender")

# Correlation matrix
weights_df.corr()

# Visualizing the correlation matrix

cmap = sns.diverging_palette(h_neg = 10,
                             h_pos = 240,
                             as_cmap=True)
sns.heatmap(weights_df.corr(), center=0,
            cmap=cmap, linewidths=1,
            annot=True, fmt=".2f")

# Visualizing the correlation matrix
corr = weights_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))


sns.heatmap(weights_df.corr(), mask=mask,
            center=0, cmap=cmap, linewidths=1,
            annot=True, fmt=".2f")

# Removing highly correlated features
# Create positive correlation matrix
corr_df = chest_df.corr().abs()
# Create and apply mask
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)
tri_df
# Find columns that meet threshold
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
print(to_drop)
# Drop those columns
reduced_df = chest_df.drop(to_drop, axis=1)

# Correlation caveats - causation
sns.scatterplot(x="N firetrucks sent to fire",
                y="N wounded by fire",data=fire_df)
```

## Capitulo 4

```python
# Feature generation - BMI
df_body['BMI'] = df_body['Weight kg'] / df_body['Height m'] ** 2
df_body.drop(['Weight kg','Height m'], axis=1)
# Feature generation - averages
leg_df['leg mm'] = leg_df[['right leg mm', 'left leg mm']].mean(axis=1)
leg_df.drop(['right leg mm', 'left leg mm'], axis=1)
```

### Intro PCA

```python
sns.scatterplot(data=df, x='handlength', y='footlength')

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# Calculating the principal components
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
std_df = scaler.fit_transform(df)

from sklearn.decomposition import PCA
pca = PCA()
print(pca.fit_transform(std_df))

# Principal component explained variance ratio
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(std_df)
print(pca.explained_variance_ratio_)


pca = PCA()
pca.fit(ansur_std_df)
print(pca.explained_variance_ratio_)

pca = PCA()
pca.fit(ansur_std_df)
print(pca.explained_variance_ratio_.cumsum())
```

#### PCA applications

```python
print(pca.components_)

# PCA in a pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pipe = Pipeline([
                  ('scaler', StandardScaler()),
                  ('reducer', PCA())])
pc = pipe.fit_transform(ansur_df)
print(pc[:,:2])

# Checking the effect of categorical features
print(ansur_categories.head())

ansur_categories['PC 1'] = pc[:,0]
ansur_categories['PC 2'] = pc[:,1]
sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='Height_class', alpha=0.4)

sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='Gender', alpha=0.4)

sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='BMI_class', alpha=0.4)

# PCA in a model pipeline
pipe = Pipeline([
                 ('scaler', StandardScaler()),
                 ('reducer', PCA(n_components=3)),
                 ('classifier', RandomForestClassifier())])
print(pipe['reducer'])

pipe.fit(X_train, y_train)
pipe['reducer'].explained_variance_ratio_

pipe['reducer'].explained_variance_ratio_.sum()

print(pipe.score(X_test, y_test))
```

#### Principal Component selection

```python
# Setting an explained variance threshold
pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', PCA(n_components=0.9))])

# Fit the pipe to the data
pipe.fit(poke_df)
print(len(pipe['reducer'].components_))

# An optimal number of components
pipe.fit(poke_df)
var = pipe['reducer'].explained_variance_ratio_
plt.plot(var)
plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()
```

## Code sniped PCA

### Utilizar PCA 

En este ejercicio, aplicarás PCA al conjunto de datos wine, para ver si puedes aumentar la precisión del modelo.

* Instanciar un objeto PCA.
* Define las características (X) y las etiquetas (y) de wine, utilizando las etiquetas de la columna "Type".
* Aplica PCA a X_train y X_test, asegurándote de que no haya fugas de datos, y almacena los valores transformados como pca_X_train y pca_X_test.
* Imprime el atributo .explained_variance_ratio_ de pca para comprobar cuánta varianza explica cada componente


```python
# Instantiate a PCA object
pca = ____()

# Define the features and labels from the wine dataset
X = wine.drop(____, ____)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply PCA to the wine dataset X vector
pca_X_train = ___.____(____)
pca_X_test = ___.____(____)

# Look at the percentage of variance explained by the different components
print(____)
```

### Entrenar un modelo con PCA

Ahora que has ejecutado PCA en el conjunto de datos wine, finalmente entrenarás un modelo KNN utilizando los datos transformados.

Ajusta el modelo knn a las características transformadas de PCA, pca_X_train, y a las etiquetas de entrenamiento, y_train.
Imprime la precisión del conjunto de pruebas del modelo knn utilizando pca_X_test y y_test.

```python
# Fit knn to the training data
____

# Score knn on the test data and print it out
____
```

## kernel PCA

* https://github.com/Sasidhar007/PCA-and-Kernel-PCA/blob/master/KernelPCA.ipynb
* https://colab.research.google.com/drive/1BRbzPRc3SKhGD7q7FVDdVuhi97eLdvjl?usp=sharing
* https://github.com/Alexander-Barth/MachineLearningNotebooks/blob/master/kernel-pca.ipynb
* https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html
* https://www.kaggle.com/code/goktugkoc/kernel-pca
* https://github.com/jeffprosise/Machine-Learning/tree/master


## Links

* https://github.com/zorzalerrante/intro_machine_learning/blob/master/05.02-Introducing-Scikit-Learn.ipynb
* https://github.com/zorzalerrante/intro_machine_learning/tree/master
* https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
* https://github.com/jbagnato/machine-learning/blob/master/Ejercicio_PCA.ipynb
* https://www.datacamp.com/es/tutorial/principal-component-analysis-in-python
* https://geostatsguy.github.io/MachineLearningDemos_Book/MachineLearning_PCA.html
* https://gitlabio.z6.web.core.windows.net/aai-url/
* https://github.com/matworx/inf-ks-notebooks
* https://www.kaggle.com/datasets/seshadrikolluri/ansur-ii/data
* https://www.kaggle.com/datasets/saifsagor/ansur-ii/data
* https://web.stanford.edu/class/stats202/notes/Unsupervised/Clustering.html
* https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.09-Principal-Component-Analysis.ipynb
* https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Principal%20Component%20Analysis.ipynb
* https://cs.colby.edu/courses/S23/cs251/projects/p4pca/p4pca251.html
* https://scikit-learn.org/1.3/auto_examples/decomposition/plot_pca_iris.html
