import pandas
import re

attributes = [
	'Finishing', 'Penalties', 'Marking', 'Age', 'Aggression', 
	'GKPositioning', 'Curve', 'GKDiving', 'Interceptions', 
	'Vision', 'Composure', 'Nationality', 'Strength', 'LongShots', 
	'SlidingTackle', 'Jumping', 'BallControl', 'Dribbling', 
	'GKKicking', 'Stamina', 'Acceleration', 'Crossing', 'GKReflexes', 
	'Agility', 'LongPassing', 'GKHandling', 'Reactions', 'ShotPower', 
	'Volleys', 'ShortPassing', 'Balance', 'StandingTackle', 'Position',
	'SprintSpeed', 'HeadingAccuracy', 'Positioning', 'Club']

# attributes we can only use for 2018-2019:
# 	ratings for that player at every position
# 	contract expiration

# TODO/improvements:
# 	how to best encode Position, Nationality, Club 

def columnNameFixer(name):
	original = name
	# some special cases
	if (name == 'Free kick accuracy' or name == 'FreekickAccuracy'):
		name = 'FKAccuracy' # align with 2019 dataset

	if (name == 'Rating'):	# specific to 2017 dataset
		return 'Overall'

	if (name == 'Heading'):
		return 'HeadingAccuracy'

	if (name == 'Speed'):
		return 'SprintSpeed'

	if (name == 'Attacking_Position'):
		return 'Positioning'

	if (name == 'Preffered_Position' or name == 'Preferred Positions'):
		name = 'Position' # align with 2019 dataset

	# if attribute is 2 words, capitalize second and join them
	sep = name.split()
	if len(sep) == 2:
		sep[1] = sep[1].capitalize()
		name = ''.join(sep)

	# strip whitespace and underscore
	name = re.sub(r'[_\s]', '', name)	

	if (name == 'LongPass' or name == 'ShortPass'):
		name += 'ing' # align with 2019 dataset

	print('{} ==> {}'.format(original, name))
	return name

print("*************** 2017 ****************")
df1 = pandas.read_csv('./data/2017.csv')
print(df1.columns)
print("\nNow renaming...")
df1 = df1.rename(columnNameFixer, axis='columns')
c1 = set(df1.columns)
print(c1)
print()
print("*************** 2018 ****************")
df2 = pandas.read_csv('./data/2018.csv')
print(df2.columns)
print("\nNow renaming...")
df2 = df2.rename(columnNameFixer, axis='columns')
c2 = set(df2.columns)
print(c2)
print()
print("*************** 2019 ****************")
df3 = pandas.read_csv('./data/2019.csv')
print(df3.columns)
print("\nNow renaming...")
df3 = df3.rename(columnNameFixer, axis='columns')
c3 = set(df3.columns)
print(c3)
print()

print(c1 & c2 & c3)
