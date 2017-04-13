#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import MySQLdb
import csv

db =  'psrvdb'
user =  'justin'
password = 'WGTo0dz9'
host = '120.26.105.20'
port = 3306

f = '%Y-%m-%d %H:%M:%S'
connection = MySQLdb.connect(host=host,user=user,passwd=password,db=db,charset="utf8")

cursor = connection.cursor()

sql = "select time,valve_pressure1 from P0 where t0_id = 20101 order by time"

cursor.execute(sql)

results = cursor.fetchall()
length = len(results)
# print("results = ",results)

head = ["time",'valve_pressure1']

filename = "../data/20101.csv"
fp = open(filename,'w')
writer = csv.writer(fp)
writer.writerow(head)
for i in range(int(length/10)):
    t = results[10*i]
    row = [t[0].strftime(f),float(t[1])]
    # print("row = ",row)
    writer.writerow(row)
fp.close()