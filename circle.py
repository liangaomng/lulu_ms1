from pycirclize import Circos
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

sectors = {"Scale 1": 1,
           "Scale 2": 2,
           "Scale_3":3,
           "Scale 4": 4,
           "Scale 5": 5}
circos = Circos(sectors, space=10)
vmin1, vmax1 = 0, 10
vmin2, vmax2 = -100, 100
for sector in circos.sectors:
    # Plot heatmap ,最外层是每个scale的贡献度
    track1 = sector.add_track((80, 100))
    track1.axis()
    track1.xticks_by_interval(1)
    data = np.random.randint(vmin1, vmax1 + 1, (1, int(sector.size)))
    track1.heatmap(data, vmin=vmin1, vmax=vmax1, show_value=True)
    # Plot heatmap with labels
    track2 = sector.add_track((50, 70))
    #track2 表示不同的激活函数
    track2.axis()
    x = np.linspace(1, int(track2.size), int(track2.size)) - 0.5
    xlabels = [str(int(v + 1)) for v in x]
    track2.xticks(x, xlabels, outer=False)
    track2.yticks([0.5, 1.5, 2.5, 3.5, 4.5], list("Phi() silu()"), vmin=0, vmax=5)
    data = np.random.randint(vmin2, vmax2 + 1, (2, int(sector.size)))
    track2.heatmap(data, vmin=vmin2, vmax=vmax2, cmap="viridis", rect_kws=dict(ec="white", lw=1))

circos.colorbar(bounds=(0.35, 0.55, 0.3, 0.01), vmin=vmin1, vmax=vmax1, orientation="horizontal")
circos.colorbar(bounds=(0.35, 0.45, 0.3, 0.01), vmin=vmin2, vmax=vmax2, orientation="horizontal", cmap="viridis")

fig = circos.plotfig()
plt.title("different scales and differnt activation functions")
plt.show()
