'''
Tools to clean, to transform and to manage data from a dataset
'''
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_df(data,
			 categorical_cols=['lead_source', 'campaign_name', 'group_source', 'group_landing', 'type', 'role'],
			 columns_scale = ["place_within_tenant", "continent", "country_code",
							  "city", "region_code", "nondst_utc_offset"]):
	'''
	Method to clean the raw dataset according to the criteria explained below, affecting categorical 
	and numerical variables. Besides it prepares the dataset with the predictors for a potential classifier.
	Criteria:
	    remove fields that are not helping for the clustering such as fakeId, signup_date or year
    	(but we will keep month as before)
		remove fields with a high percentage of missing data such as company_industrygroups, category_sectors
		and company_employes_qty.
		remove data from the initial years.
		extract month from signup_date
    	transformation of categorical features generating the alternative ones with numbers (id of the category)
    	min-max scaler for numerical values
	:param df: the raw dataframe
	:param categorical_cols: the categorical columns
	:param columns_scale: columns in the dataset to scale using minmax scaler
	:return: The cleaned dataset
	'''
	df = data.copy()
	# time date transformations
	df["signup_date"] = pd.to_datetime(df["signup_date"])
	df["year"] = df["signup_date"].dt.year
	df["month"] = df["signup_date"].dt.month
	
	print("removing non relevant features")
	# filters
	df = df[df["year"]>2013]
	df.drop(['fakeId', 'signup_date', 'year'], axis=1, inplace=True)
	df.drop(['company_industrygroups', 'category_sectors', 'company_employes_qty'], axis=1, inplace=True)

	print("processing categorical features")
	# categorical data
	for cat in categorical_cols:
		df[cat] = df[cat].astype('category')
	
	# fix campaign_name
	df.campaign_name = df.campaign_name.cat.add_categories(["other"])
	df = df.apply(lambda x: x.mask(x.map(x.value_counts())<100, "other") if x.name == "campaign_name"  else x)
	
	# fix role
	df.role = df.role.cat.add_categories(["minor"])
	df = df.apply(lambda x: x.mask(x.map(x.value_counts())<100, "minor") if x.name == "role"  else x)

	print("fixing nan values")
	# fill nan values
	df.type = df.type.cat.add_categories(["unknown"])
	df.type = df.type.fillna("unknown")
	df.role = df.role.fillna("unknown")
	df.place_within_tenant = df.place_within_tenant.fillna(0)
	columns_fix = ['continent', 'country_code', 'city', 'region_code']
	for col in columns_fix:
		df[col] = df[col].fillna(0)
		
	df.nondst_utc_offset = df.nondst_utc_offset.fillna(27)
	columns_miss = ["campaign_name", "group_source", "group_landing", "lead_source"]
	for col in columns_miss:
		df[col] = df[col].cat.add_categories(["missing"])
		df[col] = df[col].fillna("missing")

	print("scaling numerical features")
	# numerical data scaling (keep notice of the ranges for future input parameters transformation)
	for col in columns_scale:
		# the min max scaler requires a vector
		transformer = MinMaxScaler().fit(df[col].values.reshape(-1, 1)) # single feature
		transformed_data = transformer.transform(df[col].values.reshape(-1, 1))
		df[col+"_mm"] = transformed_data[:,0]
		
	for col in df.columns:
		if df[col].dtype.name == 'category':
			df[col + "_cat"] = df[col].cat.codes
		
	# selecting columns
	print("selecting final predictors")
	# get label index
	l_i = df.columns.get_loc("label")
	sel_df = df.iloc[:,l_i:]
	sel_df.drop(['nondst_utc_offset_mm', 'continent_mm', 'country_code_mm'], axis=1, inplace=True)
	return sel_df

def scale_num_features(place_within_tenant,
					   city,
					   region_code,
					   tenant_range=[1,261],
					   city_range=[0, 27332],
					   region_range=[0,391]):
	'''
	Function to scale the given numerical parameters according to min-max scaler and given the ranges for each one
	:param place_within_tenant:
	:param city:
	:param region_code:
	:param tenant_range:
	:param city_range:
	:param region_range:
	:return:
	'''
	# apply minmax scaler to the input parameters according to the provided ranges
	min, max = [0, 1]
	# Formula
	# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
	# check inputs
	if place_within_tenant < 0:
		place_within_tenant = 0
	elif place_within_tenant > tenant_range[1]:
		# if it is outside the range we assign the max accepted value
		place_within_tenant = tenant_range[1]
	if city < 0:
		city = 0
	elif city > city_range[1]:
		# if it is outside the range we assign the max accepted value
		city = city_range[1]
	if region_code < 0:
		region_code = 0
	elif region_code > region_range[1]:
		# if it is outside the range we assign the max accepted value
		region_code = region_range[1]

	# scale the three variables
	place_within_tenant_std = (place_within_tenant - tenant_range[0]) / (tenant_range[1] - tenant_range[0])
	place_within_tenant_scaled = place_within_tenant_std * (max - min) + min
	city_std = (city - city_range[0]) / (city_range[1] - city_range[0])
	city_scaled = city_std * (max - min) + min
	region_code_std = (region_code - region_range[0]) / (region_range[1] - region_range[0])
	region_code_scaled = region_code_std * (max - min) + min
	return place_within_tenant_scaled, city_scaled, region_code_scaled

def generate_cat_mapping(data,
						 categorical_cols = ['lead_source', 'campaign_name', 'group_source',
											 'group_landing', 'type', 'role']):
	'''
	Function to generate the map of the categorical values to their corresponding ids
	:param data:
	:param categorical_cols:
	:return:
	'''
	print("Transforming columns into categorical")
	df = data.copy()
	df = df[categorical_cols]
	for cat in categorical_cols:
		df[cat] = df[cat].astype('category')

	print("Reducing features with too many categories")
	# TODO this is not changing the range of the original category code
	# TODO we would need to scale this feature too with MinMax Scaler
	# fix campaign_name
	df.campaign_name = df.campaign_name.cat.add_categories(["other"])
	df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < 100, "other") if x.name == "campaign_name" else x)
	print(df.campaign_name.describe())
	# fix role
	df.role = df.role.cat.add_categories(["minor"])
	df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < 100, "minor") if x.name == "role" else x)

	# fix nan values
	print("Replacing nan values")
	# for type is going to be a new category "unknown"
	df.type = df.type.cat.add_categories(["unknown"])
	df.type = df.type.fillna("unknown")
	# for role is going to be the existing category "unknown"
	df.role = df.role.fillna("unknown")
	columns_miss = ["campaign_name", "group_source", "group_landing", "lead_source"]
	for col in columns_miss:
		df[col] = df[col].cat.add_categories(["missing"])
		df[col] = df[col].fillna("missing")

	# add codes
	print("Adding category codes")
	for col in df.columns:
		if df[col].dtype.name == 'category':
			df[col + "_cat"] = df[col].cat.codes

	#print("length before {}".format(len(df)))
	df.drop_duplicates(inplace=True)
	#print("length after {}".format(len(df)))
	print("Saving the map")
	df.to_pickle("cat_map")
	return df

def process_cat_features(lead_source, campaign_name, group_source,
       group_landing, type, role):
	'''
	Function to transform these categorical parameters into their corresponding IDs
	:param lead_source:
	:param campaign_name:
	:param group_source:
	:param group_landing:
	:param type:
	:param role:
	:return:
	'''
	try:
		# read categorical variables map
		cat_map = pd.read_pickle("cat_map")

		cat_map_ls = cat_map[['lead_source', 'lead_source_cat']].copy()
		cat_map_ls.drop_duplicates(inplace=True)
		cat_map_cn = cat_map[['campaign_name', 'campaign_name_cat']].copy()
		cat_map_cn.drop_duplicates(inplace=True)
		cat_map_gs = cat_map[['group_source', 'group_source_cat']].copy()
		cat_map_gs.drop_duplicates(inplace=True)
		cat_map_gl = cat_map[['group_landing', 'group_landing_cat']].copy()
		cat_map_gl.drop_duplicates(inplace=True)
		#print("cat_mat_gl {}".format(cat_map_gl))
		cat_map_t = cat_map[['type', 'type_cat']].copy()
		cat_map_t.drop_duplicates(inplace=True)
		cat_map_r = cat_map[['role', 'role_cat']].copy()
		cat_map_r.drop_duplicates(inplace=True)

		# try to find the names in the map
		if lead_source in cat_map_ls.lead_source.cat.categories:
			#print("found")
			lead_source_id = cat_map_ls.loc[cat_map_ls['lead_source'] == lead_source]['lead_source_cat'].values[0]
		else:
			print("the lead_source {} was not found, assigning a minor class".format(lead_source))
			lead_source_id = cat_map_ls.loc[cat_map_ls['lead_source'] == 'missing']['lead_source_cat'].values[0]

		if campaign_name in cat_map_cn.campaign_name.cat.categories:
			#print("found")
			campaign_name_id = cat_map_cn.loc[cat_map_cn['campaign_name'] == campaign_name]['campaign_name_cat'].values[0]
		else:
			print("the campaign_name {} was not found, assigning a minor class".format(campaign_name))
			campaign_name_id = cat_map_cn.loc[cat_map_cn['campaign_name'] == 'missing']['campaign_name_cat'].values[0]

		if group_source in cat_map_gs.group_source.cat.categories:
			#print("found")
			group_source_id = cat_map_gs.loc[cat_map_gs['group_source'] == group_source]['group_source_cat'].values[0]
		else:
			print("the group_source {} was not found, assigning a minor class".format(group_source))
			group_source_id = cat_map_gs.loc[cat_map_gs['group_source'] == 'missing']['group_source_cat'].values[0]

		if group_landing in cat_map_gl.group_landing.cat.categories:
			#print("found")
			group_landing_id = cat_map_gl.loc[cat_map_gl['group_landing'] == group_landing]['group_landing_cat'].values[0]
		else:
			print("the group_landing {} was not found, assigning a minor class".format(group_landing))
			group_landing_id = cat_map_gl.loc[cat_map_gl['group_landing'] == 'missing']['group_landing_cat'].values[0]

		if type in cat_map_t.type.cat.categories:
			#print("found")
			type_id = cat_map_t.loc[cat_map_t['type'] == type]['type_cat'].values[0]
		else:
			print("the type {} was not found, assigning a minor class".format(type))
			type_id = cat_map_t.loc[cat_map_t['type'] == 'unknown']['type_cat'].values[0]

		if role in cat_map_r.role.cat.categories:
			#print("found")
			role_id = cat_map_r.loc[cat_map_r['role'] == role]['role_cat'].values[0]
		else:
			print("the role {} was not found, assigning a minor class".format(role))
			role_id = cat_map_r.loc[cat_map_r['role'] == 'unknown']['role_cat'].values[0]

		return lead_source_id, campaign_name_id, group_source_id, group_landing_id, type_id, role_id
	except Exception as e:
		print(e)
		print("Error while getting the categorical ids")
		return None


if __name__ == '__main__':
	try:
		if len(sys.argv) < 1:
			print("error no data")
			sys.exit()
		filename = str(sys.argv[1])
		df = pd.read_csv(filename)
		#result = clean_df(df)
		result = generate_cat_mapping(df)
		#print(result.head)
		#print(result.columns)
	except Exception as e:
		print(e)