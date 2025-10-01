import numpy as np
from marinetools.utils import auxiliar
from scipy.interpolate import Rbf, griddata
import pandas as pd


def class_storm_seasons(df_vars_ciclos, type_: str = "WSSF"):
    """Splits the data into seasons.

    Args:
        df_vars_ciclos (pd.DataFrame): Events information.

    Returns:
        df_vars_ciclos (pd.DataFrame): Events information with a new column for the season.
    """

    df_vars_ciclos["season"] = None

    # Define season rules as a mapping for clarity and compactness
    season_defs = {
        "WSSF": {
            "winter": lambda idx: ((idx.month == 12) & (idx.day >= 21)) | (idx.month == 1) | (idx.month == 2) | ((idx.month == 3) & (idx.day < 21)),
            "spring": lambda idx: ((idx.month == 3) & (idx.day >= 21)) | (idx.month == 4) | (idx.month == 5) | ((idx.month == 6) & (idx.day < 21)),
            "summer": lambda idx: ((idx.month == 6) & (idx.day >= 21)) | (idx.month == 7) | (idx.month == 8) | ((idx.month == 9) & (idx.day < 21)),
            "fall":   lambda idx: ((idx.month == 9) & (idx.day >= 21)) | (idx.month == 10) | (idx.month == 11) | ((idx.month == 12) & (idx.day < 21)),
        },
        "WS": {
            "WS": lambda idx: ((idx.month == 12) & (idx.day >= 21)) | (idx.month == 1) | (idx.month == 2) | (idx.month == 3) | (idx.month == 4) | (idx.month == 5) | ((idx.month == 6) & (idx.day < 21)),
            "SF": lambda idx: ((idx.month == 6) & (idx.day >= 21)) | (idx.month == 7) | (idx.month == 8) | (idx.month == 9) | (idx.month == 10) | (idx.month == 11) | ((idx.month == 12) & (idx.day < 21)),
        },
        "SF": {
            "SS": lambda idx: ((idx.month == 3) & (idx.day >= 21)) | (idx.month == 4) | (idx.month == 5) | (idx.month == 6) | (idx.month == 7) | (idx.month == 8) | ((idx.month == 9) & (idx.day < 21)),
            "FW": lambda idx: ((idx.month == 9) & (idx.day >= 21)) | (idx.month == 10) | (idx.month == 11) | (idx.month == 12) | (idx.month == 1) | (idx.month == 2) | ((idx.month == 3) & (idx.day < 21)),
        },
    }

    idx = df_vars_ciclos.index
    if type_ in season_defs:
        for season, rule in season_defs[type_].items():
            mask = rule(idx)
            df_vars_ciclos.loc[mask, "season"] = season

    return df_vars_ciclos



def classification(cases, cases_sha, data, method, notrain):
    """
    Classifies data using various machine learning classifiers.

    Args:
        cases (pd.DataFrame): Training data (features).
        cases_sha (pd.DataFrame): Training data (labels).
        data (pd.DataFrame): Data to classify.
        method (str): Classifier method to use.
        notrain (int): Number of samples to use for training.

    Returns:
        np.ndarray: Classification scores or probabilities.
    """

    from sklearn import preprocessing
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1
        ),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }

    clf = classifiers[method]
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(cases_sha.iloc[:notrain, 0].values)
    clf.fit(cases.iloc[:notrain, :].values, encoded)
    # Optionally, you can compute the score on the validation set:
    # score = clf.score(cases.iloc[notrain:, :].values, cases_sha.iloc[notrain:, 'Hs'].values)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(data)
    else:
        Z = clf.predict_proba(data)[:, 1]

    return Z



def mda(data, variables, n_cases, mvar, fname="cases"):
    """
    Implements the Maximum Dissimilarity Algorithm (Camus et al. 2011).

    Args:
        data (pd.DataFrame): Raw time series.
        variables (list): Names of variables to use for dissimilarity.
        n_cases (int): Number of representative cases to select.
        mvar (str): Name of the main variable which determines the first subset.
        fname (str): Name of the file to save. Defaults to 'cases'.

    Returns:
        pd.DataFrame: The representative values of the variables.
    """

    datan = normalize(data, variables)
    n = datan.shape[0]
    ind_ = []

    # Convert to numpy array for efficiency
    X = datan[variables].values
    # If there are circular variables, adjust them
    is_circ = [v.lower().startswith('d') for v in variables]
    if any(is_circ):
        for j, circ in enumerate(is_circ):
            if circ:
                X[:, j] = np.mod(X[:, j], 2)

    # Compute distance matrix considering circular variables
    D = np.zeros((n, n))
    for j, v in enumerate(variables):
        if is_circ[j]:
            diff = np.abs(X[:, None, j] - X[None, :, j])
            D += np.minimum(diff, 2 - diff) ** 2
        else:
            diff = X[:, None, j] - X[None, :, j]
            D += diff ** 2
    D = np.sqrt(D)

    # Iterative selection
    # First point: maximum of the main variable
    first_idx = datan.loc[:, mvar].idxmax()
    ind_.append(first_idx)
    sel_pos = [datan.index.get_loc(first_idx)]

    # Initialize vector of minimum distances
    min_dist = D[sel_pos[0], :].copy()
    min_dist[sel_pos[0]] = -np.inf  # to avoid reselecting

    for _ in range(1, n_cases):
        next_pos = np.argmax(min_dist)
        ind_.append(datan.index[next_pos])
        sel_pos.append(next_pos)
        min_dist = np.minimum(min_dist, D[next_pos, :])
        min_dist[sel_pos] = -np.inf
        print("Index selected:", next_pos)


    cases = data.loc[ind_, :].copy()
    cases.insert(0, 'id', range(1, len(cases) + 1))
    cases.to_csv(fname, index=False)
    return cases


def reconstruction(
    cases_deep,
    data_deep,
    cases_shallow,
    index,
    base_vars,
    recons_vars,
    method="rbf-multiquadric",
    smooth=0.5,
    optimize=True,
    optimizer="local",
    eps=1.0,
    num=100,
    scale_data=True,
    scaler_method="StandardScaler"):
    """
    Reconstructs deep water variables from shallow water data using regression methods.

    This function uses relationships established between deep and shallow water data
    to reconstruct missing deep water variables based on available base variables.

    Args:
        cases_deep (pd.DataFrame): Representative cases with deep water data (training X)
        data_deep (pd.DataFrame): Deep water data to reconstruct (prediction X)
        cases_shallow (pd.DataFrame): Corresponding shallow water cases (training Y)
        index (pd.Index): Index for the reconstructed data
        base_vars (list): Names of base variables used for reconstruction
        recons_vars (list): Names of variables to be reconstructed
        method (str, optional): Regression method. Defaults to 'rbf-multiquadric'.
            Options: 'linear', 'nearest', 'cubic', 'rbf-*', 'gp-*'
        smooth (float, optional): Smoothing parameter for RBF. Defaults to 0.5.
        optimize (bool, optional): Whether to optimize RBF epsilon. Defaults to True.
        num (int, optional): Number of training samples to use (randomly selected). Defaults to 80% of available data.
        scale_data (bool, optional): If False, data will not be scaled. Defaults to True.
        scaler_method (str, optional): Scaling method for normalization. Defaults to 'StandardScaler'.

    Returns:
        pd.DataFrame: Reconstructed deep water data with variables in recons_vars

    Example:
        >>> reconstructed = reconstruction(
        ...     cases_deep=deep_cases,
        ...     data_deep=deep_data,
        ...     cases_shallow=shallow_cases,
        ...     index=deep_data.index,
        ...     base_vars=['Hs', 'Tp'],
        ...     recons_vars=['U10', 'Dir']
        ... )
    """

    # Extract base variables from all sets
    base_train = cases_deep[base_vars].copy()  # Representative cases (training)
    base_pred = data_deep[base_vars].copy()    # Data to reconstruct (prediction)
    target_train = cases_shallow[recons_vars].copy()  # Variables to reconstruct (training)

    # Adjust num to 80% of training data size if None or out of bounds
    n_train = base_train.shape[0]
    if num is None or num >= n_train or num < 2:
        num = int(0.8 * n_train)
        if num < 1:
            num = 1
        if num >= n_train:
            num = n_train - 1

    if scale_data:
        # Normalize base variables using the same scaler
        _, base_scaler = auxiliar.scaler(cases_shallow[base_vars], method=scaler_method)
        base_train_norm, _ = auxiliar.scaler(base_train, scale=base_scaler, method=scaler_method)
        base_pred_norm, _ = auxiliar.scaler(base_pred, scale=base_scaler, method=scaler_method)
    else:
        base_train_norm = target_train.values if hasattr(target_train, 'values') else target_train
        base_train_norm = base_train.values if hasattr(base_train, 'values') else base_train
        base_pred_norm = base_pred.values if hasattr(base_pred, 'values') else base_pred

    # Initialize output DataFrame
    data_reconstructed = pd.DataFrame(index=index, columns=recons_vars)

    # Reconstruir cada variable objetivo de forma independiente

    for target_var in recons_vars:
        # Asegurar que target_train sea DataFrame (2D) para evitar errores en sklearn
        target_col = cases_shallow[[target_var]].copy() if target_var in cases_shallow else pd.DataFrame(cases_shallow[target_var].copy())
        if scale_data:
            target_train_norm, target_scaler = auxiliar.scaler(target_col, method=scaler_method)
        else:
            target_train_norm = target_col.values if hasattr(target_col, 'values') else target_col

        # Regresión en espacio normalizado o no
        target_pred_norm = regression(
            base_train=base_train_norm,
            target_train=target_train_norm,
            base_pred=base_pred_norm,
            method=method,
            num=num,
            smooth=smooth,
            optimize=optimize,
            optimizer=optimizer,
            eps=eps,
        )

        # Desnormalizar predicciones si corresponde
        if scale_data:
            target_pred, _ = auxiliar.scaler(
                target_pred_norm.reshape(-1, 1),
                transform=False,
                scale=target_scaler,
                method=scaler_method,
            )
            data_reconstructed[target_var] = target_pred.flatten()
        else:
            data_reconstructed[target_var] = target_pred_norm.flatten()

    return data_reconstructed


def regression(
    base_train, target_train, base_pred, method="rbf-multiquadric", num=100, smooth=1, optimize=True, eps=1, optimizer="local"
):
    """
    Performs regression using various interpolation and machine learning methods.
    
    This function supports multiple regression approaches including interpolation
    methods (linear, cubic, nearest), radial basis functions (RBF), and 
    Gaussian processes (GP).
    
    Args:
        X (pd.DataFrame or np.ndarray): Training input features (predictors)
        Y (pd.Series or np.ndarray): Training target values (response)
        x (pd.DataFrame or np.ndarray): Test input features for prediction
        method (str, optional): Regression method. Defaults to 'rbf-multiquadric'.
            Available methods:
            - Interpolation: 'linear', 'nearest', 'cubic'
            - RBF: 'rbf-multiquadric', 'rbf-inverse', 'rbf-gaussian', 
                   'rbf-linear', 'rbf-cubic', 'rbf-quintic', 'rbf-thin_plate'
            - Gaussian Process: 'gp-rbf', 'gp-exponential', 'gp-quadratic', 'gp-white'
        num (int, optional): Number of points for RBF optimization. Defaults to 100.
        smooth (float, optional): Smoothing parameter for RBF methods. Defaults to 1.
        optimize (bool, optional): Whether to optimize RBF epsilon parameter. Defaults to True.
        param (float, optional): Manual epsilon parameter for RBF (used if optimize=False). Defaults to 1.
    
    Returns:
        np.ndarray: Predicted values for input x
        
    Raises:
        ValueError: If the specified method is not implemented
        
    Example:
        >>> predictions = regression(
        ...     X=train_features,
        ...     Y=train_targets,
        ...     x=test_features,
        ...     method='rbf-multiquadric',
        ...     optimize=True
        ... )
    """
    
    # Import required libraries
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
    )
    
    # Available methods
    available_methods = [
        "linear", "nearest", "cubic",
        "rbf-multiquadric", "rbf-inverse", "rbf-gaussian", 
        "rbf-linear", "rbf-cubic", "rbf-quintic", "rbf-thin_plate",
        "gp-rbf", "gp-exponential", "gp-quadratic", "gp-white",
    ]
    
    # Gaussian Process kernels
    gp_kernels = {
        "gp-rbf": 1.0 * RBF(1.0),
        "gp-exponential": ExpSineSquared(),
        "gp-quadratic": RationalQuadratic(),
        "gp-white": WhiteKernel(),
    }
    
    # Convertir a numpy arrays si es necesario
    if hasattr(base_train, 'values'):
        base_train = base_train.values
    if hasattr(target_train, 'values'):
        target_train = target_train.values
    if hasattr(base_pred, 'values'):
        base_pred = base_pred.values

    # Asegurar que los arrays sean 2D
    if base_train.ndim == 1:
        base_train = base_train.reshape(-1, 1)
    if base_pred.ndim == 1:
        base_pred = base_pred.reshape(-1, 1)

    # Validar dimensiones
    if base_train.shape[0] != len(target_train):
        raise ValueError(f"base_train y target_train deben tener el mismo número de muestras. base_train: {base_train.shape}, target_train: {len(target_train)}")
    if base_train.shape[1] != base_pred.shape[1]:
        raise ValueError(f"base_train y base_pred deben tener el mismo número de variables. base_train: {base_train.shape[1]}, base_pred: {base_pred.shape[1]}")
    
    try:
        # Method 1: Scipy interpolation methods
        if method in ["linear", "nearest", "cubic"]:
            predictions = griddata(base_train, target_train, base_pred, method=method)
            
        # Method 2: Radial Basis Function (RBF) methods
        elif method.startswith("rbf-"):
            rbf_function = method.split("-")[1]

            # Optimizar epsilon si se solicita
            if optimize:
                base_train = base_train + np.random.normal(0, 1e-8, base_train.shape)
                params = auxiliar.optimize_rbf_epsilon(
                    base_train, target_train, num, method=rbf_function, smooth=smooth, eps0=eps, optimizer=optimizer
                )
                epsilon, smooth = params
            else:
                epsilon = eps

            coords = [base_train[:, i] for i in range(base_train.shape[1])]
            coords.append(target_train)
            rbf_ = Rbf(*coords, function=rbf_function, smooth=smooth, epsilon=epsilon)
            pred_coords = [base_pred[:, i] for i in range(base_pred.shape[1])]
            predictions = rbf_(*pred_coords)
            
        # Method 3: Gaussian Process methods
        elif method.startswith("gp-"):
            kernel = gp_kernels[method]

            # Crear y ajustar Gaussian Process
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=False
            )
            gp.fit(base_train, target_train)

            # Predecir (solo media)
            predictions = gp.predict(base_pred, return_std=False)
            
        else:
            raise ValueError(
                f"Method '{method}' is not implemented. "
                f"Available methods: {available_methods}"
            )
                    
        return predictions
        
    except Exception as e:
        # Fallback to linear interpolation if the chosen method fails
        print(f"Warning: {method} failed ({str(e)}), falling back to linear interpolation")
        try:
            return griddata(base_train, target_train, base_pred, method="linear")
        except:
            # Último recurso: devolver la media
            print("Warning: Linear interpolation also failed, returning mean values")
            return np.full(len(base_pred), np.mean(target_train))



def normalize(data, variables, circular=False):
    """Normalizes data using the maximum distance between values

    Args:
        * data (pd.DataFrame): raw time series

    Returns:
        * datan (pd.DataFrame): normalized variable
    """

    datan = data.copy()
    for i in variables:
        if i.startswith("Dir"):
            circular = True
        
        if circular:
            datan[i] = np.deg2rad(data[i]) / np.pi
        else:
            datan[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

    return datan