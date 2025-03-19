# Tarea 2


## t-SNE

### Fragmentos de codigo

```py

# Data t-SNE on female ANSUR dataset
df.shape

non_numeric = ['BMI_class', 'Height_class', 'Gender', 'Component', 'Branch']
df_numeric = df.drop(non_numeric, axis=1)
df_numeric.shape


# Model
from sklearn.manifold import TSNE

m = TSNE(learning_rate=50)
tsne_features = m.fit_transform(df_numeric)
tsne_features[1:4,:]
```

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
