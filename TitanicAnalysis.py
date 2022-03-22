import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import numpy.random
from pandas.tools.plotting import scatter_matrix, radviz
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
# eclipse da errore ma in realta' funziona ...
from patsy import dmatrices
from sklearn import datasets, svm, tree, preprocessing
import sklearn.ensemble as ske
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from IPython.display import Image


df = pd.read_csv("C:/Users/601787621/workspace/data/titanic3.csv", delimiter=";")

df = df.drop(['ticket','cabin'], axis=1)
# Remove NaN values
#df = df.dropna()
df["age"].fillna(int(df.age.mean()), inplace=True)

print(df.corr()['survived'])

'''
# DECISION TREES
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()
# Convert Sex variable to numeric
encoded_sex = label_encoder.fit_transform(df["sex"])
# Initialize model
tree_model = tree.DecisionTreeClassifier()
# Train the model
tree_model.fit(X = pd.DataFrame(encoded_sex), y = df["survived"])

with open("C:/Users/601787621/workspace/data/tree1.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["survived"], out_file=f)

# with the graphviz library installed, convert the file in this way
# dot tree1.dot -Tpng -o tree1.png


preds = tree_model.predict_proba(X = pd.DataFrame(encoded_sex))

print(pd.crosstab(preds[:,0], df["sex"]))
print("Survival guess rate",tree_model.score(X = pd.DataFrame(encoded_sex), y = df["survived"]))


# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, df["pclass"]]).T
# Train the model
tree_model.fit(X = predictors, y = df["survived"])

with open("C:/Users/601787621/workspace/data/tree2.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["Sex", "Class"], out_file=f)

preds = tree_model.predict_proba(X = predictors)

# Create a table of predictions by sex and class
print(pd.crosstab(preds[:,0], columns = [df["pclass"],df["sex"]]))
print("Survival guess rate",tree_model.score(X = predictors, y = df["survived"]))



predictors = pd.DataFrame([encoded_sex,df["pclass"],df["age"],df["fare"]]).T

# Initialize model with maximum tree depth set to 8
tree_model = tree.DecisionTreeClassifier(max_depth = 8)
tree_model.fit(X = predictors, y = df["survived"])

with open("C:/Users/601787621/workspace/data/tree3.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, 
                              feature_names=["sex", "class","age","fare"], 
                              out_file=f)
'''




'''
# model formula
# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our 
# regression know that those variables are categorical.
# Ref: http://patsy.readthedocs.org/en/latest/formulas.html
formula = 'survived ~ C(pclass) + C(sex) + age + sibsp  + C(embarked)' 
# create a results dictionary to hold our regression results for easy analysis later        
results = {} 

# create a regression friendly dataframe using patsy's dmatrices function
y,x = dmatrices(formula, data=df, return_type='dataframe')
# instantiate our model
model = sm.Logit(y,x)
# fit our model to the training data
res = model.fit()
# save the result for outputing predictions later
results['Logit'] = [res, formula]
res.summary()

# Plot Predictions Vs Actual
plt.figure(figsize=(18,4));
plt.subplot(121, axisbg="#DBDBDB")
# generate predictions from our fitted model
ypred = res.predict(x)
plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
plt.grid(color='white', linestyle='dashed')
plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');

# Residuals
ax2 = plt.subplot(122, axisbg="#DBDBDB")
plt.plot(res.resid_dev, 'r-')
plt.grid(color='white', linestyle='dashed')
ax2.set_xlim(-1, len(res.resid_dev))
plt.title('Logit Residuals');
plt.show()

fig = plt.figure(figsize=(18,9))
a = .2
# Below are examples of more advanced plotting. 
# It it looks strange check out the tutorial above.
fig.add_subplot(221, axisbg="#DBDBDB")
kde_res = KDEUnivariate(res.predict())
kde_res.fit()
plt.plot(kde_res.support,kde_res.density)
plt.fill_between(kde_res.support,kde_res.density, alpha=a)
plt.title("Distribution of our Predictions")

fig.add_subplot(222, axisbg="#DBDBDB")
plt.scatter(res.predict(),x['C(Sex)[T.male]'] , alpha=a)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of survival")
plt.ylabel("Gender Bool")
plt.title("The Change of Survival Probability by Gender (1 = Male)")

fig.add_subplot(223, axisbg="#DBDBDB")
plt.scatter(res.predict(),x['C(Pclass)[T.3]'] , alpha=a)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

fig.add_subplot(224, axisbg="#DBDBDB")
plt.scatter(res.predict(),x.Age , alpha=a)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Survival Probability by Age")
plt.xlabel("Predicted chance of survival")
plt.ylabel("Age")
plt.show()


# test the model on other data ...
test_data = pd.read_csv("data/titanic3_test.csv", delimiter=";")
test_data['survived'] = 1.23

# Use your model to make prediction on our test set. 
compared_results = ka.predict(test_data, results, 'Logit')
compared_results = Series(compared_resuts)  # convert our model to a series for easy output
print (compared_results)
'''





'''
fig, ax = plt.subplots()
df.survived.value_counts().plot(kind='barh', color="blue", alpha=.65)
ax.set_ylim(-1, len(df.survived.value_counts()))
plt.title("Survival Breakdown (1 = Survived, 0 = Died)")
plt.show()


fig = plt.figure(figsize=(18,6))
#create a plot of two subsets, male and female, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph.
#'barh' is just a horizontal bar graph
df_male = df.survived[df.sex == 'male'].value_counts().sort_index()
df_female = df.survived[df.sex == 'female'].value_counts().sort_index()
print(df_male)

ax1 = fig.add_subplot(121)
df_male.plot(kind='bar',label='Male', alpha=0.55)
df_female.plot(kind='bar', color='#FA2379',label='Female', alpha=0.55)
plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')
ax1.set_ylim(-1, 600)

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(122)
(df_male/float(df_male.sum())).plot(kind='bar',label='Male', alpha=0.55)
(df_female/float(df_female.sum())).plot(kind='bar', color='#FA2379',label='Female', alpha=0.55)
plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')
ax2.set_ylim(0, 1)
plt.show()
'''



# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6))
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
df.survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.survived, df.age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
df.pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(df.pclass.value_counts()))
plt.title("Class Distribution")

ax4 = plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
df.age[df.pclass == 1].plot(kind='kde')
df.age[df.pclass == 2].plot(kind='kde')
df.age[df.pclass == 3].plot(kind='kde')
# plots an axis label
plt.xlabel("Age")
#ax4.set_xlim([0,100])
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

ax5 = plt.subplot2grid((2,3),(1,2))
df.embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(df.embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")
plt.show()



fig = plt.figure(figsize=(18,4))
alpha_level = 0.65

# building on the previous code, here we create an additional subset with in the gender subset 
# we created for the survived variable. I know, thats a lot of subsets. After we do that we call 
# value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 
# class pair.
ax1=fig.add_subplot(141)
female_highclass = df.survived[df.sex == 'female'][df.pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)
ax1.set_xticklabels(["survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
female_lowclass = df.survived[df.sex == 'female'][df.pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
male_lowclass = df.survived[df.sex == 'male'][df.pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
male_highclass = df.survived[df.sex == 'male'][df.pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')
plt.show()


fig = plt.figure(figsize=(18,12))
a = 0.65

# Step 2
ax2 = fig.add_subplot(341)
df.survived[df.sex == 'male'].value_counts().plot(kind='bar',label='Male')
df.survived[df.sex == 'female'].value_counts().plot(kind='bar', color='#FA2379',label='Female')
ax2.set_xlim(-1, 2)
plt.title("Step. 2 \nWho Survied? with respect to Gender."); plt.legend(loc='best')

ax3 = fig.add_subplot(342)
(df.survived[df.sex == 'male'].value_counts()/float(df.sex[df.sex == 'male'].size)).plot(kind='bar',label='Male')
(df.survived[df.sex == 'female'].value_counts()/float(df.sex[df.sex == 'female'].size)).plot(kind='bar', color='#FA2379',label='Female')
ax3.set_xlim(-1,2)
plt.title("Who Survived proportionally?"); plt.legend(loc='best')


# Step 3
ax4 = fig.add_subplot(343)
female_highclass = df.survived[df.sex == 'female'][df.pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=a)
ax4.set_xticklabels(["survived", "Died"], rotation=0)
ax4.set_xlim(-1, len(female_highclass))
plt.title("Who survived? with respect to Gender and Class"); plt.legend(loc='best')
plt.show()



formula_ml = 'survived ~ C(pclass) + C(sex) + age + sibsp + parch + C(embarked)'

'''
# set plotting parameters
plt.figure(figsize=(8,6))


# create a regression friendly data frame
y, x = dmatrices(formula_ml, data=df, return_type='matrix')

# select which features we would like to analyze
# try chaning the selection here for diffrent output.
# Choose : [2,3] - pretty sweet DBs [3,1] --standard DBs [7,3] -very cool DBs,
# [3,6] -- very long complex dbs, could take over an hour to calculate! 
feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:,[feature_1, feature_2]]  


y = np.asarray(y)
# needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape. 
y = y.flatten()      

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

# do a cross validation
nighty_precent_of_sample = int(.9 * n_sample)
X_train = X[:nighty_precent_of_sample]
y_train = y[:nighty_precent_of_sample]
X_test = X[nighty_precent_of_sample:]
y_test = y[nighty_precent_of_sample:]

# create a list of the types of kerneks we will use for your analysis
types_of_kernels = ['linear', 'rbf', 'poly']

# specify our color map for plotting the results
color_map = plt.cm.RdBu_r

# fit the model
for fig_num, kernel in enumerate(types_of_kernels):
    clf = svm.SVC(kernel=kernel, gamma=3)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

    # circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.show()
'''

'''
#RANDOM FORESTS
formula_ml = 'survived ~ C(pclass) + C(sex) + age + sibsp + parch + C(embarked)'

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
# RandomForestClassifier expects a 1 dimensional NumPy array, so we convert
y = np.asarray(y).ravel()
#instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
score = results_rf.score(x, y)
print ("Mean accuracy of Random Forest Predictions on the data was: {0}".format(score))
'''


'''
dtype={"year":int,
       "stint":int,
       "GP":int,
       "GS":int,
       "minutes":int,
       "points":int,
       "oRebounds":int,
       "dRebounds":int,
       "rebounds":int,
       "assists":int,
       "steals":int,
       "blocks":int,
       "turnovers":int,
       "PF":int,
       "fgAttempted":int,
       "fgMade":int,
       "ftAttempted":int,
       "ftMade":int,
       "threeAttempted":int,
       "threeMade":int,
       "PostGP":int,
       "PostGS":int,
       "PostMinutes":int,
       "PostPoints":int,
       "PostoRebounds":int,
       "PostdRebounds":int,
       "PostRebounds":int,
       "PostAssists":int,
       "PostSteals":int,
       "PostBlocks":int,
       "PostTurnovers":int,
       "PostPF":int,
       "PostfgAttempted":int,
       "PostfgMade":int,
       "PostftAttempted":int,
       "PostftMade":int,
       "PostthreeAttempted":int,
       "PostthreeMade":int}

df = pd.read_csv("C:/Users/601787621/workspace/data/basketball_players.csv", dtype)
'''
