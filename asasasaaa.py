from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# suppose you have trained a classifier clf
predicted_labels = clf.predict(x_data)

# now to get confusion matrix one can use the confusion_matrix
# utility provided by sklearn
conf_mat = confusion_matrix(true_labels, predicted_labels)

# now it's time to plot the matrix
# some standard matplotlib code
# add a subplot
fig = plt.figure()
ax = fig.add_subplot(111)

# plot the matrix
cax = ax.matshow(conf_mat)

# add colorbar for reference
fig.colorbar(cax)

# add labels to plot
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("Conf_Mat.jpg")
plt.show()