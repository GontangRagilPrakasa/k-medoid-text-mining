from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app import app
#from flask import Flask


#app = Flask(__name__)
db = SQLAlchemy(app)
migrate = Migrate(app, db)


class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, unique=True, primary_key=True, nullable=False)
    name = db.Column(db.String, nullable=False)
    medoid = db.Column(db.String, nullable=True)
    clusters = db.relationship('Cluster', backref='dataset', lazy='dynamic')

    def __repr__(self):
        return "<Datasets: {}>".format(self.name)

    def __init__(self, name):
        self.name = name



class Cluster(db.Model):
    __tablename__ = 'cluster'
    id = db.Column(db.Integer, unique=True, primary_key=True, nullable=False)
    name = db.Column(db.String, nullable=False)
    number_of_cluster = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    cluster_data = db.relationship('ClusterData', backref='cluster_detail', lazy='dynamic')

    def _repr_(self):
        return "<Name: {}>".format(self.name)

    def __init__(self, name, number_of_cluster, dataset_id):
        self.name = name
        self.number_of_cluster = number_of_cluster
        self.dataset_id = dataset_id

class ClusterData(db.Model):
    __tablename__ = 'cluster_data'
    id = db.Column(db.Integer, unique=True, primary_key=True, nullable=False)
    title = db.Column(db.String, nullable=False)
    cluster = db.Column(db.Integer, nullable=False)
    cluster_id = db.Column(db.Integer, db.ForeignKey('cluster.id'))

    def _repr_(self):
        return "<Title: {}>".format(self.title)

    def __init__(self, title, cluster, cluster_id):
        self.title = title
        self.cluster = cluster
        self.cluster_id = cluster_id