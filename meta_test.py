from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

print('\nExecuting random seed')
tpot = TPOT(generations=5, verbosity=2, population_size=5, seed_method='random')
tpot.fit(X_train, y_train)

print('\nExecuting generator seed')
tpot = TPOT(generations=5, verbosity=2, population_size=5, seed_method='generator')
tpot.fit(X_train, y_train)

print('\nExecuting pooled seed')
tpot = TPOT(generations=5, verbosity=2, population_size=5, seed_method='pool')
tpot.fit(X_train, y_train)


