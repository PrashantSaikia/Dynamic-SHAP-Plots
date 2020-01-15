from shap_plots import shap_summary_plot, shap_dependence_plot
import plotly.tools as tls
import dash_core_components as dcc
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost
import shap
import matplotlib
import plotly.graph_objs as go
try:
    import matplotlib.pyplot as pl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
except ImportError:
    pass
from sklearn import preprocessing 

cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
}  # #1E88E5 -> #ff0052
red_blue = LinearSegmentedColormap('RedBlue', cdict1)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

red_blue = matplotlib_to_plotly(red_blue, 255)

def summary_plot_plotly_fig(dataset, target='target column', max_display = 20):
    data = pd.read_csv(dataset, encoding="ISO-8859-1")
    X = data.drop(['target column'], axis=1)

    y = data[target]
    y = y/max(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    X_train.fillna((-999), inplace=True) 
    X_test.fillna((-999), inplace=True)

    _, shap_values, feature_names = train_model_and_return_shap_values(X, y, target)

    mpl_fig = shap_summary_plot(shap_values, pd.DataFrame(X_train, columns=X.columns), feature_names=feature_names, max_display=20)

    plotly_fig = tls.mpl_to_plotly(mpl_fig)

    plotly_fig['layout'] = {'xaxis': {'title': 'SHAP value (impact on model output)'}}

    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    text = [feature_names[i] for i in feature_order]
    text = iter(text)

    for i in range(1, len(plotly_fig['data']), 2):
        t = text.__next__()
        plotly_fig['data'][i]['name'] = ''
        plotly_fig['data'][i]['text'] = t
        plotly_fig['data'][i]['hoverinfo'] = 'text'

    colorbar_trace  = go.Scatter(x=[None],
                                 y=[None],
                                 mode='markers',
                                 marker=dict(
                                     colorscale=red_blue, 
                                     showscale=True,
                                     cmin=-5,
                                     cmax=5,
                                     colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
                                 ),
                                 hoverinfo='none'
                                )

    plotly_fig['layout']['showlegend'] = False
    plotly_fig['layout']['hovermode'] = 'closest'
    plotly_fig['layout']['height']=600
    plotly_fig['layout']['width']=500

    plotly_fig['layout']['xaxis'].update(zeroline=True, showline=True, ticklen=4, showgrid=False)
    plotly_fig['layout']['yaxis'].update(dict(visible=False))
    plotly_fig.add_trace(colorbar_trace)
    plotly_fig.layout.update(
                             annotations=[dict(
                                  x=1.18,
                                  align="right",
                                  valign="top",
                                  text='Feature value',
                                  showarrow=False,
                                  xref="paper",
                                  yref="paper",
                                  xanchor="right",
                                  yanchor="middle",
                                  textangle=-90,
                                  font=dict(family='Calibri', size=14)
                                )
                             ],
                             margin=dict(t=20)
                            )
    return plotly_fig

def train_model_and_return_shap_values(X, y, target):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    X_train.fillna((-999), inplace=True) 
    X_test.fillna((-999), inplace=True)

    # Some of values are float or integer and some object. This is why we need to cast them:
    for f in X_train.columns: 
        if X_train[f].dtype=='object':
            lbl = preprocessing.LabelEncoder() 
            lbl.fit(list(X_train[f].values)) 
            X_train[f] = lbl.transform(list(X_train[f].values))

    for f in X_test.columns: 
        if X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder() 
            lbl.fit(list(X_test[f].values)) 
            X_test[f] = lbl.transform(list(X_test[f].values))

    X_train=np.array(X_train) 
    X_test=np.array(X_test) 
    X_train = X_train.astype(float) 
    X_test = X_test.astype(float)

    d_train = xgboost.DMatrix(X_train, label=y_train, feature_names=list(X))
    d_test = xgboost.DMatrix(X_test, label=y_test, feature_names=list(X))

    # train the model
    params = {
        "eta": 0.01,
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "silent": 1
    }

    model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=None, early_stopping_rounds=50)
    feature_names = model.feature_names
    shap_values = shap.TreeExplainer(model).shap_values(pd.DataFrame(X_train, columns=X.columns))
    return model, shap_values, feature_names

def dependence_plot_to_plotly_fig(dataset, target='target column', max_display=10):
    data = pd.read_csv(dataset, encoding="ISO-8859-1")
    X = data.drop(['target column'], axis=1)
    y = data[target]
    y = y/max(y)

    xgb_full = xgboost.DMatrix(X, label=y)

    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_test = xgboost.DMatrix(X_test, label=y_test)

    # use validation set to choose # of trees
    params = {
        # "eta": 0.002,
        # "max_depth": 3,
        # "subsample": 0.5,
        "silent": 1
    }
    model_train = xgboost.train(params, xgb_train, 3000, evals = [(xgb_test, "test")], verbose_eval=None)

    # train final model on the full data set
    params = {
        # "eta": 0.002,
        # "max_depth": 3, 
        # "subsample": 0.5,
        "silent": 1
    }
    model = xgboost.train(params, xgb_full, 1500, evals = [(xgb_full, "test")], verbose_eval=None)
    features = model.feature_names
    shap_values = shap.TreeExplainer(model).shap_values(X)

    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    features = [features[i] for i in feature_order[::-1]]

    lis = []
    for i in features:
        mpl_fig, interaction_index = shap_dependence_plot(i, shap_values, X)
        plotly_fig = tls.mpl_to_plotly(mpl_fig)

        # The x-tick labels start by default from 0, which is not necessarily the min value of the feature.
        # So, we need to increment the x-tick labels by 1. But while doing so, the y-axis gets shifted.
        # To prevent that, we need to manually control the x-axis range from r_min to r_max
        new_x = []
        for j in plotly_fig['data'][0]['x']:
            new_x.append(j)

        r_min = min(plotly_fig['data'][0]['x'])
        r_max = max(plotly_fig['data'][0]['x'])

        plotly_fig['layout']['xaxis'].update(range=[r_min-1, r_max+1])
        plotly_fig['data'][0]['x'] = tuple(new_x)

        # Define the colorbar
        colorbar_trace  = go.Scatter(x=[None],
                                     y=[None],
                                     mode='markers',
                                     marker=dict(
                                         colorscale=red_blue, 
                                         showscale=True,
                                         colorbar=dict(thickness=5, outlinewidth=0),
                                         color=[min(X[X.columns[interaction_index]]), max(X[X.columns[interaction_index]])],
                                     ),
                                     hoverinfo='none'
                                    )

        plotly_fig['layout']['showlegend'] = False
        plotly_fig['layout']['hovermode'] = 'closest'
        plotly_fig['layout']['height']=380
        plotly_fig['layout']['width']=450
        plotly_fig['layout']['xaxis'].update(zeroline=True, 
                                             showline=True, 
                                             ticklen=4, 
                                             showgrid=False,
                                             tickmode='linear')
        title = plotly_fig['layout']['yaxis']['title']
        plotly_fig['layout']['yaxis'].update(title=title.split(' -')[0])

        plotly_fig.add_trace(colorbar_trace)
        plotly_fig.layout.update(
                                 annotations=[dict(
                                      x=1.23,
                                      align="right",
                                      valign="top",
                                      text=X.columns[interaction_index],
                                      showarrow=False,
                                      xref="paper",
                                      yref="paper",
                                      xanchor="right",
                                      yanchor="middle",
                                      textangle=-90,
                                      font=dict(family='Calibri', size=14)
                                    )
                                 ],
                                 margin=dict(t=50, b=50, l=50, r=80)
                                )
        lis.append(plotly_fig)
    return lis, features

def interaction_plot_to_plotly_fig(dataset, target_col='target column', max_display=10):
    data = pd.read_csv(dataset, encoding="ISO-8859-1")
    X = data.drop(['target column'], axis=1)
    y = data[target_col]
    y = y/max(y)

    xgb_full = xgboost.DMatrix(X, label=y)

    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_test = xgboost.DMatrix(X_test, label=y_test)

    # use validation set to choose # of trees
    params = {
        # "eta": 0.002,
        # "max_depth": 3,
        # "subsample": 0.5,
        "silent": 1
    }
    model_train = xgboost.train(params, xgb_train, 3000, evals = [(xgb_test, "test")], verbose_eval=None)

    # train final model on the full data set
    params = {
        # "eta": 0.002,
        # "max_depth": 3, 
        # "subsample": 0.5,
        "silent": 1
    }
    model = xgboost.train(params, xgb_full, 1500, evals = [(xgb_full, "test")], verbose_eval=None)
    features = model.feature_names
    shap_values = shap.TreeExplainer(model).shap_values(X)

    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    features = [features[i] for i in feature_order[::-1]]

    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X)

    lis = []
    for i in features:
        for j in features:
            mpl_fig = pl.figure()
            ax = mpl_fig.add_subplot(111)
            _, interaction_index = shap_dependence_plot ( (i, j), shap_interaction_values, X.iloc[:2000,:] )
            plotly_fig = tls.mpl_to_plotly(mpl_fig)

            r_min = min(plotly_fig['data'][0]['x'])
            r_max = max(plotly_fig['data'][0]['x'])

            plotly_fig['layout']['xaxis'].update(range=[r_min-1, r_max+1])
            plotly_fig['layout']['showlegend'] = False
            plotly_fig['layout']['hovermode'] = 'closest'
            plotly_fig['layout']['height']=380
            plotly_fig['layout']['width']=450
            plotly_fig['layout']['xaxis'].update(zeroline=True, 
                                                 showline=True, 
                                                 ticklen=4, 
                                                 showgrid=False,
                                                 tickmode='linear')
            plotly_fig['layout']['yaxis'].update(showline=True)

            if i!=j:
                # plotly_fig['layout']['height']=380
                plotly_fig['layout']['width']=480
                plotly_fig['layout']['yaxis']['title'] = "SHAP interaction value for {} and {}".format(i.split('-')[0], j.split('-')[0])
                # Define the colorbar
                colorbar_trace = go.Scatter(x=[None],
                                            y=[None],
                                            mode='markers',
                                            marker=dict(
                                                colorscale=red_blue, 
                                                showscale=True,
                                                colorbar=dict(thickness=5, outlinewidth=0),
                                                color=[min(X[X.columns[interaction_index]]), max(X[X.columns[interaction_index]])],
                                            ),
                                            hoverinfo='none'
                                           )
                plotly_fig.add_trace(colorbar_trace)
                plotly_fig.layout.update(
                                         annotations=[dict(
                                              x=1.23,
                                              align="right",
                                              valign="top",
                                              text=X.columns[interaction_index],
                                              showarrow=False,
                                              xref="paper",
                                              yref="paper",
                                              xanchor="right",
                                              yanchor="middle",
                                              textangle=-90,
                                              font=dict(family='Calibri', size=14)
                                            )
                                         ],
                                         margin=dict(t=30, b=30, l=60, r=80)
                                        )
            else:
                plotly_fig['layout']['yaxis']['title'] = "SHAP main effect value for {}".format(i.split('-')[0])
            lis.append(plotly_fig)
    return lis, features
