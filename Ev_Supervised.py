from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from tmp_PA import *

# Evaluation quantitative des résultats "supervisé" : Gabriel Lefèvre


def make_conf_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))
    aff_matrice = pd.DataFrame(matrix, index=labels, columns=labels)
    return aff_matrice


def calcul_precision(dataf_mat):
    matrix = dataf_mat.values
    vrai_positif = matrix.diagonal()
    faux_positif = matrix.sum(axis=0) - vrai_positif
    faux_negatif = matrix.sum(axis=1) - vrai_positif

    precision = vrai_positif / (vrai_positif + faux_positif)
    rappel = vrai_positif / (vrai_positif + faux_negatif)

    return precision, rappel


def courbe_ROC(str_model):
    model = None
    if str_model == "SVM":
        model = SVC(kernel='linear', C=1.0, probability=True)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)
    binary_y_test = label_binarize(y_test, classes=np.unique(y_test))
    nb_ofclass = binary_y_test.shape[1]
    fpos = dict()
    tpos = dict()
    roc = dict()
    for i in range(nb_ofclass):
        fpos[i], tpos[i], ths = roc_curve(binary_y_test[:, i], pred_proba[:, i].ravel())
        roc[i] = auc(fpos[i], tpos[i])
    # print(fpos, tpos, ths, roc)
    fpos, tpos, _ = roc_curve(binary_y_test.ravel(), pred_proba.ravel())
    roc["macro"] = auc(fpos, tpos)
    plt.plot(fpos, tpos, label='Classe {}'.format(nb_ofclass))

    # Paramètres du graphique
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC par classe')
    plt.legend(loc='lower right')
    plt.show()


print("------SVM-Test------")
_, pred = classification_SVM()
matrix_ofconfusion = make_conf_matrix(y_test, pred)
# print(matrix_ofconfusion)
precision, rappel = calcul_precision(matrix_ofconfusion)
print("Précision = ", precision)
print("Rappel = ", rappel)
courbe_ROC("SVM")
print("-------Random Forest-Test------")
_, pred = classification_Random_Forest()
matrix_ofconfusion = make_conf_matrix(y_test, pred)
precision, rappel = calcul_precision(matrix_ofconfusion)
print("Précision = ", precision)
print("Rappel = ", rappel)
print("-------Multilayer Perceptron-Test------")
_, pred = classification_MLP()
matrix_ofconfusion = make_conf_matrix(y_test, pred)
precision, rappel = calcul_precision(matrix_ofconfusion)
print("Précision = ", precision)
print("Rappel = ", rappel)
#
