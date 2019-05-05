import pandas
import re
from collections import defaultdict

# these features exist in all 3 datasets
features = [
	'Finishing', 'Penalties', 'Marking', 'Age', 'Aggression', 
	'GKPositioning', 'Curve', 'GKDiving', 'Interceptions', 
	'Vision', 'Composure', 'Strength', 'LongShots', 
	'SlidingTackle', 'Jumping', 'BallControl', 'Dribbling', 
	'GKKicking', 'Stamina', 'Acceleration', 'Crossing', 'GKReflexes', 
	'Agility', 'LongPassing', 'GKHandling', 'Reactions', 'ShotPower', 
	'Volleys', 'ShortPassing', 'Balance', 'StandingTackle',
	'SprintSpeed', 'HeadingAccuracy', 'Positioning']

# these exists in all 3 but we haven't encoded them properly yet 
# to use in model
unencoded = [
	'Club', 'Nationality', 'Position'
]

# attributes we can only use for 2018-2019:
# 	ratings for that player at every position
# 	contract expiration

# TODO/improvements for later:
# 	how to use Category Encoders for 'Position', 'Nationality', 'Club' 

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

	return name

def fetchAndClean(dataPath):
	df = pandas.read_csv(dataPath)
	return df.rename(columnNameFixer, axis='columns')

def addIDs(df2017, lookup):
	# imperfect matching, but the approach is to consider two players equal if
	# they have the same first initial, last name, nationality.
	for index, row in df2017.iterrows():
		try:
			firstInitial = row.Name.split()[0][0]
			lastName = row.Name.split()[-1]
			nationality = row.Nationality
		except:
			continue

		key = frozenset([firstInitial, lastName, nationality])
		if key in lookup:
			df2017.at[index, 'ID'] = lookup[key]

	return df2017

# build a map from player key (first initial, last name, nationality)
# to set of IDs (multiple IDs problem handles outside this function)
def buildLookup(df, lookup=defaultdict(set)):
	lookup = defaultdict(set)
	for _, row in df.iterrows():
		try:
			firstInitial = row.Name.split()[0][0]
			lastName = row.Name.split()[-1]
			nationality = row.Nationality
		except:
			print('excpetion')
			continue

		key = frozenset([firstInitial, lastName, nationality])
		lookup[key].add(row.ID)

	return lookup

def prune(df2017, df2018, df2019):
	commonIDs = set(df2017['ID']) & set(df2018['ID']) & set(df2019['ID'])
	print('Num common IDs: ', len(commonIDs))
	df2017 = df2017.loc[df2017['ID'].isin(commonIDs)]
	df2018 = df2018.loc[df2018['ID'].isin(commonIDs)]
	df2019 = df2019.loc[df2019['ID'].isin(commonIDs)]
	return df2017, df2018, df2019

df2017 = fetchAndClean('./data/2017.csv')
df2018 = fetchAndClean('./data/2018.csv')
df2019 = fetchAndClean('./data/2019.csv')

print("2017 shape: ", df2017.shape)
print("2018 shape: ", df2018.shape)
print("2019 shape: ", df2019.shape)

# necessary because 2017 dataset doesn't include IDs
lookup = buildLookup(df2018)
lookup = buildLookup(df2019, lookup)

# disambiguation -- throw out any players matched with 
# multiple IDs (this could happen because our heuristic
# for matching players is initial, last name, nationality)
cleaned = {}
for key, idSet in lookup.items():
	if len(idSet) > 1: continue
	cleaned[key] = idSet.pop()

df2017.insert(0, 'ID', -1) # add ID column with all set to -1
df2017 = addIDs(df2017, cleaned)

withIDs2017 = df2017.loc[df2017['ID'] != -1]
print('Num IDs added to 2017 dataset ', withIDs2017.shape[0])

# remove any players that don't exist in all 3 datasets
print("Removing players/features that aren't present in all 3 datasets...")
df2017, df2018, df2019 = prune(withIDs2017, df2018, df2019)

features2017 = set(df2017.columns)
features2018 = set(df2018.columns)
features2019 = set(df2019.columns)

commonFeatures = features2017 & features2018 & features2019
for unencodedFeature in unencoded:
	try:
		commonFeatures.remove(unencodedFeature)
	except KeyError as e:
		print('KeyError: ', e)

commonFeatures = list(commonFeatures)
df2017 = df2017[commonFeatures]
df2018 = df2018[commonFeatures]
df2019 = df2019[commonFeatures]
print("pruned 2017 shape: ", df2017.shape)
print("pruned 2018 shape: ", df2018.shape)
print("pruned 2019 shape: ", df2019.shape)
print('Features: ', commonFeatures)

# df2017.to_csv('./data/2017CommonFeatures.csv')
# df2018.to_csv('./data/2018CommonFeatures.csv')
# df2019.to_csv('./data/2019CommonFeatures.csv')
