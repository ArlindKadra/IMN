import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 35,
        'axes.titlesize': 35,
        'axes.labelsize': 35,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'legend.fontsize': 25,
    },
    style="white"
)

base_inn = 0.9049

inn_removing_features = [0.88688, 0.89509, 0.89863, 0.90216, 0.9044]

base_catboost = 0.928

catboost_removing_features = [0.90795, 0.93255, 0.928, 0.92459, 0.93265]

base_decision_tree = 0.744

decision_tree_removing_features = [0.7561, 0.7507, 0.7531, 0.71871, 0.75172]

base_tabnet = 0.90796

tabnet_removing_features = [0.91126, 0.90315, 0.89087, 0.91232, 0.91078]

base_shap = 0.9057
shap_removing_features = [0.90654, 0.89958, 0.90193, 0.8954, 0.8954]


inn_improvement_fraction = []
for i in range(len(inn_removing_features)):
    inn_improvement_fraction.append((base_inn - inn_removing_features[i]) / base_inn)

catboost_improvement_fraction = []
for i in range(len(catboost_removing_features)):
    catboost_improvement_fraction.append((base_catboost - catboost_removing_features[i]) / base_catboost)

decision_tree_improvement_fraction = []
for i in range(len(decision_tree_removing_features)):
    decision_tree_improvement_fraction.append((base_decision_tree - decision_tree_removing_features[i]) / base_decision_tree)

tabnet_improvement_fraction = []
for i in range(len(tabnet_removing_features)):
    tabnet_improvement_fraction.append((base_tabnet - tabnet_removing_features[i]) / base_tabnet)

shap_improvement_fraction = []
for i in range(len(shap_removing_features)):
    shap_improvement_fraction.append((base_shap - shap_removing_features[i]) / base_shap)

inn_improvement_fraction = np.array(inn_improvement_fraction)
catboost_improvement_fraction = np.array(catboost_improvement_fraction)
decision_tree_improvement_fraction = np.array(decision_tree_improvement_fraction)
tabnet_improvement_fraction = np.array(tabnet_improvement_fraction)
inn_improvement_fraction *= 100
catboost_improvement_fraction *= 100
decision_tree_improvement_fraction *= 100
tabnet_improvement_fraction *= 100
shap_improvement_fraction = np.array(shap_improvement_fraction)
shap_improvement_fraction *= 100


plt.bar([i - 0.2 for i in (1, 2, 3, 4, 5)], inn_improvement_fraction, 0.1, label='INN')
plt.bar([i - 0.1 for i in (1, 2, 3, 4, 5)], catboost_improvement_fraction, 0.1, label='CatBoost')
plt.bar([1, 2, 3, 4, 5], tabnet_improvement_fraction, 0.1, label='TabNet')
plt.bar([i + 0.1 for i in (1, 2, 3, 4, 5)], decision_tree_improvement_fraction, 0.1, label='Decision Tree')
plt.bar([i + 0.2 for i in (1, 2, 3, 4, 5)], shap_improvement_fraction, 0.1, label='SHAP')
plt.xlabel('Without top-k')
plt.xticks([1, 2, 3, 4, 5])
plt.ylabel(r'Percentual decrease in AUROC')
plt.legend()#bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=5)
plt.savefig('importance_gain.pdf', bbox_inches='tight')



