import numpy as np
import matplotlib.pyplot as plt

n_groups = 12
mse = (0.503, 0.456, 0.554, 0.356, 0.404, 0.609, 0.339, 0.338, 0.532, 0.231, 0.309, 0.428)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
rect = plt.bar(index, mse, bar_width, alpha = opacity, color = 'b', label = 'DS3A')

plt.xlabel('Ref Combination',fontsize = 18)
plt.ylabel('RMSE',fontsize = 18)
plt.title('RMSE with Different Reference Combination',fontsize = 18)
plt.xticks(index + bar_width,('Ref.1','Ref.2','Ref.3','Ref.4','Ref.5','Ref.1,2','Ref.2,4','Ref.4,5','Ref.1,2,3','Ref.1,3,5','Ref.1,2,3,4,5','No-Ref'), rotation = 300, fontsize = 16)
plt.ylim(0.0,1.0)
plt.legend()

plt.tight_layout()
plt.show()
