from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy 
from flask_marshmallow import Marshmallow 
from sqlalchemy_utils import ScalarListType
import os

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)

# Usser model
class User(db.Model):
    __tablename__ = 'user'
    email = db.Column(db.String, primary_key=True)
    name = db.Column(db.String(100))
    password = db.Column(db.String(100))
    financial_goal = db.Column(db.Float)
    time_period = db.Column(db.Integer)
    current_progress = db.Column(db.Float)
    #friends = db.relationship('Friend', backref='user', cascade='all, delete, delete-orphan', single_parent=True, order_by='desc(Friend.friend_id)')

# class Friend(db.Model):
#     __tablename__ = 'friend'
#     friend_id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.String, db.ForeignKey('user.email'))
#     name = db.Column(db.String)


def __init__(self, email, name, password, financial_goal, time_period, current_progress): #friends):
    self.name = name
    self.description = description
    self.price = price
    self.qty = qty
   # self.friends = friends


# user schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('email', 'name', 'password', 'financial_goal', 'time_period', 'current_progress', 'friends')

user_schema = UserSchema()
users_schema = UserSchema(many=True)

# Create a user
@app.route('/user', methods=['POST'])
def add_user():
  name = request.json['name']
  email = request.json['email']
  password = request.json['password']
  financial_goal = request.json['financial_goal']
  time_period = request.json['time_period']
  current_progress = request.json['current_progress']
  #friends = request.json['friends']


  new_user = User(email=email, name=name, password=password, financial_goal=financial_goal, time_period=time_period, current_progress=current_progress)

  db.session.add(new_user)
  db.session.commit()

  return user_schema.jsonify(new_user)

# Get All users
@app.route('/user', methods=['GET'])
def get_users():
  all_users = User.query.all()
  result = users_schema.dump(all_users)
  return jsonify(result)

# Get Single users
@app.route('/user/<id>', methods=['GET'])
def get_user(id):
  user = User.query.get(id)
  return user_schema.jsonify(user)

# Update a user
@app.route('/user/<id>', methods=['PUT'])
def update_user(id):
  user = User.query.get(id)

  name = request.json['name']
  email = request.json['email']
  password = request.json['password']
  financial_goal = request.json['financial_goal']
  time_period = request.json['time_period']
  current_progress = request.json['current_progress']
  #friends = request.json['friends']

  user.name = name
  user.email = email
  user.password = password
  user.financial_goal = financial_goal
  user.time_period = time_period
  user.current_progress = current_progress
  #user.friends = friends

  db.session.commit()

  return user_schema.jsonify(user)

# Delete user
@app.route('/user/<id>', methods=['DELETE'])
def delete_user(id):
  user = User.query.get(id)
  db.session.delete(user)
  db.session.commit()

  return user_schema.jsonify(user)

# Run Server
if __name__ == '__main__':
  app.run(debug=True)

