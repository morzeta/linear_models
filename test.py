import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels import IV2SLS

from ydata_profiling import ProfileReport
from preparation import read_data, get_x_y
from utils import set_random_seed
from feature_selection import recursive_elimination
from pipelines import *
from visualize import correlation_subplots, correlation_heatmap

# set seed for reproducible results
set_random_seed(1)

# read and prepare data
data = read_data()
x, y = get_x_y(data)

# ivmod = IV2SLS(y,  x, x)
# res_2sls = ivmod.fit()
# res_2sls.wu_hausman()

# inspect correlations
# correlation_heatmap(data)
# correlation_subplots(data)
# plt.show()

# data profile
# profile = ProfileReport(x, title="Profiling Report", explorative=True)
# profile.to_file("report.html")

# recursive_elimination("homogeneity_score")
# ordinary_linear_regression_v1()
# ordinary_linear_regression_v2()
# ordinary_linear_regression_v3()
linear_regression_v1()
# linear_regression_v2()


print("hi")
