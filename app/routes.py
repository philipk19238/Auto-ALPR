from app import app
from flask import render_template, jsonify, request, make_response, redirect, send_file, url_for, session
from db import *
from license_plate import * 
from werkzeug.utils import secure_filename
from api import *


@app.route('/', methods=['GET', 'POST'])
def landing():
	return render_template('landing_page.html')

@app.route('/sign_up')
def sign_up():
	return render_template('signup.html')

@app.route('/map', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		user_id = request.get_json()['id']
		return make_response(jsonify({'message':user_id}), 200)
	else:
		location_data = create_geoJSON()
		locations = get_locations()
		first_name = request.cookies['first_name']
		return render_template('main.html', geo=location_data, locations=locations, name=first_name)

@app.route('/confirm/<_id>', methods=['GET','POST'])
def confirm(_id):
	name = request.cookies['first_name']
	details = User(_id).read_profile()
	return render_template('confirm.html', details=(details), name=name)

@app.route('/model/<_id>', methods=['GET', 'POST'])
def model(_id):
	if request.method == 'POST':
		if 'video' not in request.files:
			return redirect(request.url)
		file = request.files['video']
		if file.filename == '' or not allowed_file(file.filename):
			return redirect(request.url)
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			license_plate = request.values.get('license')
			video = app.config['UPLOAD_FOLDER'] +'/' + filename
			image_location = process_video(video, license_plate, _id)
			return send_file(image_location)
	return 'Success'

@app.route('/add_complex', methods=['GET', 'POST'])
def add_complex():
	if request.method == 'POST':
		dic = request.values
		apartment = dic['apartment']
		num_spots = int(dic['spots'])
		description = dic['description']
		user_id = ObjectId()
		session['id'] = str(user_id)

		post_dic = {key:value for key,value in zip(['_id','Name', 'Description', 'num_spots'],
											       [user_id, apartment, description, num_spots])}
		user = User(str(user_id)).create_user(post_dic)
		return redirect(url_for('.add_complex_location', _id=str(user_id)))

	return render_template('add_complex.html')

@app.route('/add_complex_location/<_id>', methods=['GET', 'POST'])
def add_complex_location(_id):
	return render_template('add_complex_location.html')

@app.route('/geoconvert', methods=['GET', 'POST'])
def geoconvert():
	if request.method == 'POST':
		_id = session['id']
		dic = request.values
		address = dic['address']
		city = dic['city']
		state = dic['state']
		zip_code = dic['zip']
		latitude, longitude = get_latLng(address, city, state, zip_code)
		if longitude:
			user = User(_id)
			user.update_profile('Longitude', longitude)
			user.update_profile('Latitude', latitude)
	return 'Success'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



