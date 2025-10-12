import plotly.express as px
import numpy as np

fish_features = np.load("data/fish_features.npy")
fish_class = np.load("data/fish_class.npy")

feature_names = ["length", "weight"]
class_names = ["tuna", "salmon"]


tuna_index = np.where(fish_class == 0)[0]
salmon_index = np.where(fish_class == 1)[0]

tuna_length = fish_features[tuna_index, 0]
salmon_length = fish_features[salmon_index, 0]

tuna_weight = fish_features[tuna_index, 1]
salmon_weight = fish_features[salmon_index, 1]

# labels={"species_id": "Species", "sepal_width": "Sepal Width", "sepal_length": "Sepal Length", "petal_width": "Petal Width", "petal_length": "Petal Length", }

fig = px.parallel_coordinates(fish_features,
                              color=fish_class,
                              # labels=labels,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()
# fig.write_image("parallel_coordinate_plot.pdf")
