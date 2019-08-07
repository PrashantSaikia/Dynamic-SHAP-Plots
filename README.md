# Dynamic SHAP Plots
Enabling interactive plotting of the visualizations from the [SHAP](https://github.com/slundberg/shap) project.

Most of the code is taken from the SHAP library, with minor modifications (like returning the matplotlib figure objects instead of plotting them, changes in the directory structure, etc). The file `dynamic_shap_plots.py` binds them all together to produce the interactive visualizations with the Plotly library.

## Requirements:
```
shap
plotly
pandas
sklearn
matplotlib
xgboost
iml
scipy
numpy
```
This package has been built and tested on Windows 10 with Python 3.5. Slight modifications may be needed in case of errors when using in Linux or Mac OS.

##  Some dynamic SHAP visualizations in Jupyter notebook:

### 1. Summary Plot:
```
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from dynamic_shap_plots import summary_plot_plotly_fig as sum_plot
import warnings
warnings.filterwarnings('ignore')

plotly_fig = sum_plot(r'path\to\dataset.csv', target='target column')

init_notebook_mode(connected=True)
iplot(plotly_fig, show_link=False)
```

![](https://user-images.githubusercontent.com/39755678/62591715-16cbb480-b903-11e9-818f-82ce793af4b1.png)

To save the figure:
```
plot(plotly_fig, show_link=False, filename=r'path\to\save\figure.html')
```

### 2. Dependence Plot:
```
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from dynamic_shap_plots import dependence_plot_to_plotly_fig as dep_plot
from shap_plots import shap_summary_plot, shap_dependence_plot
import warnings
warnings.filterwarnings('ignore')

lis, features = dep_plot(r'path\to\dataset.csv', target='target column', max_display=20)

init_notebook_mode(connected=True)
for i in range(len(lis)):
  iplot(lis[i], show_link=False)
```
Alternately, you can also plot for specific features:

```
>>> features.index('Q2FC - Timeliness of billing notices/statements')
15
>>> iplot(lis[15], show_link=False)
```

![](https://user-images.githubusercontent.com/39755678/62591656-e257f880-b902-11e9-8a44-d5f75ad2304e.png)

### 3. Interaction Plot:
```
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from dynamic_shap_plots import dependence_plot_to_plotly_fig as dep_plot
from dynamic_shap_plots import interaction_plot_to_plotly_fig as int_plot
from shap_plots import shap_summary_plot, shap_dependence_plot
import warnings
warnings.filterwarnings('ignore')

lis, features = int_plot(r'path\to\dataset.csv', target='target column', max_display=20)

init_notebook_mode(connected=True)
for i in range(len(lis)):
  iplot(lis[i], show_link=False)
```
Alternately, you can also plot for specific features:

```
>>> features.index('QCF - Caring company')
262
>>> iplot(lis[262], show_link=False)
```

![](https://user-images.githubusercontent.com/39755678/62591749-39f66400-b903-11e9-9400-5c0eaec4c35d.png)
