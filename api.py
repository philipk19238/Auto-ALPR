import requests 
from bson.objectid import ObjectId
import pandas as pd
from geojson import FeatureCollection, Feature, Point
from db import * 

KEY = 'QKA92JSaGKSBVatQhWcWlATOIRH2YkEY'

class User:
	def __init__(self, _id):
		self._id = ObjectId(_id)

	def read_profile(self):
		query = collection.find({'_id':self._id})
		data = [i for i in query][0]
		data['_id'] = str(data['_id'])
		return data

	def update_profile(self, field, vals):
		collection.update_one({'_id':self._id}, {'$set':{field:vals}})

	def create_user(self, post):
		collection.insert_one(post)

#helper functions
def get_locations():
	query = collection.find()
	data = [i for i in query]
	for dic in data:
		dic['_id'] = str(dic['_id'])
	return data

def create_geoJSON():
	features = []
	dic_list = get_locations()
	for dic in dic_list:
		if 'Latitude' in dic or 'Longitude' in dic:
			longitude, latitude = map(float, (dic['Longitude'], dic['Latitude']))
			features.append(
				Feature(
					geometry = Point((longitude, latitude)),
					properties = {
						'name': dic['Name'],
						'description': dic['Description'],
						'_id': dic['_id'],
						'spots': dic['num_spots']
					}
				)
			)
	return FeatureCollection(features)

def get_latLng(address, city, state, zip_code):
	web = f"""http://www.mapquestapi.com/geocoding/v1/address?key={KEY}
			  &street={address}&city={city}&state={state}&postalCode={zip_code}"""
	response = requests.get(web).json()
	if response['results']:
		lat_lng = response['results'][0]['locations'][0]['latLng']
		return lat_lng['lat'], lat_lng['lng']
	else:
		return False, False
		
if __name__ == "__main__":
    df = pd.DataFrame(collection.find())
    if len(df) > 10:
    	print(df.tail())
    else:
    	print(df)
