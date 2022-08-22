import pyodbc
# import mysql.connector 
# from mysql.connector import errorcode
def dbConnect():
    # ServerName = "18.216.22.185"
    # MySQLDatabase = "fantasy_game"
    # username = "root"
    # password = "Umbrella#1234dev"

    ServerName = "localhost"
    MySQLDatabase = "game_of_11"
    username = "root"
    password = "C+*rbon@123"

    # try:
    # conn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={0}; database={1};UID={2};PWD={3}".format(ServerName, MSQLDatabase, username,password))
    
    conn = pyodbc.connect("DRIVER={{MySQL ODBC 8.0 Driver}};SERVER={0}; database={1};UID={2};PWD={3}".format(ServerName, MySQLDatabase, username,password))
 
    return conn

def getData(query):

    # query = getData("SELECT * FROM `players`")
    cur = dbConnect().cursor()
    cur.execute(query)
    data = cur.fetchall()
    print(data)
    return data


# query = getData("SELECT * FROM `players` WHERE 1")

# print(query)

