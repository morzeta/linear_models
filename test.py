import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
from preparation import read_data
from utils import set_random_seed

# set seed for reproducible results
set_random_seed(1)

# read and prepare data
x, y = read_data()

# data normalisation
# x = (x-x.mean())/x.std()

# correlation matrix
# sns.heatmap(x.corr(), annot=True, linewidths=.4)
# plt.title('Heatmap of correlation between variables', fontsize=16)
# plt.show()

# data profile
profile = ProfileReport(x, title="Profiling Report", explorative=True)
profile.to_file("report.html")
