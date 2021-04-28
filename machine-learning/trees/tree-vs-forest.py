from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def single_tree_scores():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0, max_depth=3)
    clf = clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_test, y_test)
    return scores


def random_forest_scores():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    clf = RandomForestClassifier(criterion="gini", random_state=0, max_depth=4)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_test, y_test)
    return scores


single = single_tree_scores()
random = random_forest_scores()

print("A Single Tree scored: {}%".format(single.mean() * 100))
print("A Forest scored: {}%".format(random.mean() * 100))
print("A difference of {}%!".format(round((random.mean() - single.mean()) * 100, 2)))

print(single)
print(random)
plt.bar([1, 2, 3, 4, 5], random * 100, color="green", label="Forest", alpha=0.8)
plt.bar([1, 2, 3, 4, 5], single * 100, color="brown", label="Single Tree", alpha=0.8, edgecolor="black", linewidth=5)

plt.xlabel("Test")
plt.ylabel("Accuracy")
plt.title("One Tree vs A Forest")
plt.legend()
#plt.savefig("trees.png")
plt.show()


# # Uncomment me for a timer!
# import time
# start_timer = time.perf_counter()
# single = single_tree_scores()
# end_timer = time.perf_counter()
# full_time = round(end_timer - start_timer, 2)
# print(f"A Single tree takes this long to compute: {full_time}")
#
# start_timer = time.perf_counter()
# random = random_forest_scores()
# end_timer = time.perf_counter()
# full_time = round(end_timer - start_timer, 2)
# print(f"A Forest takes this long to compute: {full_time}")
