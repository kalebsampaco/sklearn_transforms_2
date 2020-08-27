from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

def norm(df):
    from sklearn.preprocessing import Normalizer
    from sklearn.preprocessing import normalize
    #Quito los datos que no se van a normalizar
    part1 = df.iloc[:,[0]]
    part2 = df.iloc[:,[13]]

    # selecciono los datos a normalizar y uso la clase de normalizacion de sklearn
    transformer = Normalizer().fit_transform(df.iloc[:,1:13]) 
    transformer

    # Concatenacin de los datos
    df_data_conc1 = np.concatenate([part1, transformer, part2], axis=1)

    #Datos ya normalizados
    df_data_3 = pd.DataFrame(df_data_conc1, columns = ['USER_ID', 'HOURS_DATASCIENCE', 'HOURS_BACKEND', 'HOURS_FRONTEND',
           'NUM_COURSES_BEGINNER_DATASCIENCE', 'NUM_COURSES_BEGINNER_BACKEND',
           'NUM_COURSES_BEGINNER_FRONTEND', 'NUM_COURSES_ADVANCED_DATASCIENCE',
           'NUM_COURSES_ADVANCED_BACKEND', 'NUM_COURSES_ADVANCED_FRONTEND',
           'AVG_SCORE_DATASCIENCE', 'AVG_SCORE_BACKEND', 'AVG_SCORE_FRONTEND',
           'PROFILE'])

    return df_data_3
    
def transform_df(df):
    from sklearn.impute import SimpleImputer
    # Crear un objeto simpleimputer para llenar con ceros
    si_cero = SimpleImputer(
        missing_values=np.nan,  # los valores que faltan son del tipo nan (Pandas estándar)
        strategy='constant',  # la estrategia elegida es cambiar el valor faltante por una constante
        fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
        verbose=0,
        copy=True
    )
    # Crear un objeto ``SimpleImputer`` para llenar con el promedio
    si_mean = SimpleImputer(strategy='most_frequent', copy=True)
    
    # Llenado los nan por ceros en la sección indicada
    df1 = si_cero.fit_transform(X=df.iloc[:, 0:10])
    # Llenando los nan por la media en la sección indicada
    df2 = si_mean.fit_transform(X=df.iloc[:, 10:14])
    #Concatenación de las dos secciones transformadas
    df_data_conc = np.concatenate([df1, df2], axis=1)
    df_data_conc

    df_data_3 = pd.DataFrame(df_data_conc, columns = ['USER_ID', 'HOURS_DATASCIENCE', 'HOURS_BACKEND', 'HOURS_FRONTEND',
           'NUM_COURSES_BEGINNER_DATASCIENCE', 'NUM_COURSES_BEGINNER_BACKEND',
           'NUM_COURSES_BEGINNER_FRONTEND', 'NUM_COURSES_ADVANCED_DATASCIENCE',
           'NUM_COURSES_ADVANCED_BACKEND', 'NUM_COURSES_ADVANCED_FRONTEND',
           'AVG_SCORE_DATASCIENCE', 'AVG_SCORE_BACKEND', 'AVG_SCORE_FRONTEND',
           'PROFILE'])

    df_data_3 = norm(df_data_3)
    
    return df_data_3
