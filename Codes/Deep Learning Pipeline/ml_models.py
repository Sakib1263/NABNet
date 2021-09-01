class SVMRegressor:
    # Support Vector Machine (SVM)
    def __init__(self, train_features, test_features, labels):
        self.train_features = train_features
        self.test_features = test_features
        self.labels = labels

    # SVR
    def svr(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        #
        model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # NuSVR
    def nu_svr(self):
        from sklearn.svm import NuSVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        #
        model = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Linear SVR
    def linear_svr(self):
        from sklearn.svm import LinearSVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        #
        model = make_pipeline(StandardScaler(), LinearSVR(random_state=0))
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score


class NNeighbors:
    # Nearest Neighbor Regressors
    def __init__(self, train_features, test_features, labels):
        self.train_features = train_features
        self.test_features = test_features
        self.labels = labels

    # K-Nearest Neighbor (KNN) Regressor
    def knn_regressor(self):
        from sklearn.neighbors import KNeighborsRegressor
        #
        model = KNeighborsRegressor(n_neighbors=15)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Radius Neighbour (RNN) Regressor
    def rnn_regressor(self):
        from sklearn.neighbors import RadiusNeighborsRegressor
        #
        model = RadiusNeighborsRegressor(radius=10)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score


class Ensemble:
    # Ensemble based Regressors
    def __init__(self, train_features, test_features, labels):
        self.train_features = train_features
        self.test_features = test_features
        self.labels = labels

    # Gradient Tree Boosting Regressor
    def gradboost_regressor(self):
        from sklearn.ensemble import GradientBoostingRegressor
        #
        model = GradientBoostingRegressor(random_state=0)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Histogram based Gradient Tree Boosting Regressor
    def histgrad_boost_regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        #
        model = HistGradientBoostingRegressor()
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # ADABoost
    def adaboost_regressor(self):
        from sklearn.ensemble import AdaBoostRegressor
        #
        model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=0, loss='linear')
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # XgBoost
    def xgboost_regressor(self):
        import xgboost as xgb
        #
        model = xgb.XGBRegressor(objective='reg:tweedie')
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Bagging Regressor
    def bagging_regressor(self):
        from sklearn.svm import SVR
        from sklearn.ensemble import BaggingRegressor
        #
        model = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Random Forest Regressor
    def random_forest_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        #
        model = RandomForestRegressor(max_depth=2, random_state=0)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Extra Trees Regressor
    def extra_trees_regressor(self):
        from sklearn.ensemble import ExtraTreesRegressor
        #
        model = ExtraTreesRegressor(n_estimators=100, random_state=0)
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Voting Regressor
    def voting_regressor(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import VotingRegressor
        #
        r1 = LinearRegression()
        r2 = RandomForestRegressor(n_estimators=10, random_state=1)
        model = VotingRegressor([('lr', r1), ('rf', r2)])
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score

    # Stacking Regressor
    def stacking_regressor(self):
        from sklearn.linear_model import RidgeCV
        from sklearn.svm import LinearSVR
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import SGDRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import StackingRegressor
        #
        estimators = [('GBR', GradientBoostingRegressor(random_state=0))]
        model = StackingRegressor(estimators=estimators, final_estimator=MLPRegressor(hidden_layer_sizes=(100,),
                 activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='adaptive', max_iter=1000))
        model = model.fit(self.train_features, self.labels.ravel())
        preds = model.predict(self.test_features)
        score = model.score(self.train_features, self.labels.ravel())

        return preds, score


def logistic_regressor(train_features, test_features, labels):
    # Stochastic Gradient Descent (SGD)
    from sklearn.linear_model import LogisticRegression
    #
    model = LogisticRegression(random_state=0)
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def sgd_regressor(train_features, test_features, labels):
    # Stochastic Gradient Descent (SGD)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    #
    model = make_pipeline(StandardScaler(), SGDRegressor(loss='huber', penalty='l2', epsilon=0.06))
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def decision_tree_regressor(train_features, test_features, labels):
    # Decision Tree Regressor
    from sklearn.tree import DecisionTreeRegressor
    #
    model = DecisionTreeRegressor(criterion='mse', splitter='best')
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def pls_regressor(train_features, test_features, labels):
    # PLS Regressor
    from sklearn.cross_decomposition import PLSRegression
    #
    model = PLSRegression(n_components=50, algorithm='svd')
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def theilsen_regressor(train_features, test_features, labels):
    # TheilSen Regressor
    from sklearn.linear_model import TheilSenRegressor
    #
    model = TheilSenRegressor(random_state=0)
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def gpr_regressor(train_features, test_features, labels):
    # Gaussian Process Regressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    #
    kernel = DotProduct() + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score


def mlp_regressor(train_features, test_features, labels):
    # (Neural Network) Multi-layer Perceptron Regressor
    from sklearn.neural_network import MLPRegressor
    #
    model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001,
                         batch_size='auto', learning_rate='constant', max_iter=10000)
    model = model.fit(train_features, labels.ravel())
    preds = model.predict(test_features)
    score = model.score(train_features, labels.ravel())

    return preds, score
