from datetime import timedelta, date
from firebase import firebase
from django.shortcuts import render

#fbase_connection_cursor = firebase.FirebaseApplication(
           # 'https://umbrella-game-of-11.firebaseio.com/', None)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('/home/forhad/Study/GameOF11/umbrella-game-of-11-959e51706fb0.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://umbrella-game-of-11.firebaseio.com'
})

# As an admin, the app has access to read and write all data, regradless of Security Rules
ref = db.reference('restricted_access/secret_document')
print(ref.get())
